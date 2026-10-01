"""Autoregressive, set-based dispatcher policy.

Encoder: per-train tokens (window features + an "already cleared this step" flag) and one global
token (episode-level features + clearance index) through a small transformer encoder. The global
token keeps an empty window well defined and carries the value estimate.

Per clearance n = 0..B-1:
1. pointer over window slots, masked to choosable trains not yet cleared this step;
2. action head over {PROCEED, HOLD, YIELD_TO, REROUTE}, masked by the executor's action mask;
3. for YIELD_TO, a second pointer (query from the chosen train, keys from the window), masked to
   the train's legal conflict partners;
4. a Bernoulli continue head, only when another clearance is allowed and possible.
The joint log-probability of a decision point is the sum over these choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_flatland.tada.executor import N_ACTIONS, PROCEED, YIELD_TO

NEG = -1e9

# PyTorch's inference fast path for TransformerEncoder returns NaN when every train token is padded
# (an empty window, as at the final bootstrap state), even though the global token is not. The
# regular path is exact, and at these sizes the speed difference is negligible.
torch.backends.mha.set_fastpath_enabled(False)


class DispatcherNet(nn.Module):
    def __init__(self, n_feat: int, n_glob: int, d: int = 64, heads: int = 4, layers: int = 1):
        super().__init__()
        self.d = d
        self.tok = nn.Sequential(nn.Linear(n_feat + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(n_glob + 1, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ptr = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.act = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, N_ACTIONS))
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.cont = nn.Sequential(nn.Linear(2 * d + N_ACTIONS, d), nn.Tanh(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        # prior: follow the plan (P(PROCEED) ~ 0.87 when all four actions are legal), and stop after
        # one clearance. A train is in the window at almost every step, so a weaker prior makes the
        # untrained dispatcher edit plans hundreds of times per episode and start well below the executor.
        with torch.no_grad():
            self.act[-1].bias.zero_()
            self.act[-1].bias[PROCEED] = 3.0
            self.cont[-1].bias.fill_(-1.0)

    def encode(self, feats, mask, glob, chosen, n_frac):
        """feats (B,M,F), mask (B,M), glob (B,G), chosen (B,M), n_frac (B,) -> g (B,d), H (B,M,d)."""
        x = self.tok(torch.cat([feats, chosen.unsqueeze(-1)], -1))
        g = self.glob(torch.cat([glob, n_frac.unsqueeze(-1)], -1)).unsqueeze(1)
        tokens = torch.cat([g, x], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), mask], 1) < 0.5
        h = self.enc(tokens, src_key_padding_mask=pad)
        return h[:, 0], h[:, 1:]

    def pointer_logits(self, g, H, mask):
        ge = g.unsqueeze(1).expand_as(H)
        return self.ptr(torch.cat([H, ge], -1)).squeeze(-1).masked_fill(mask < 0.5, NEG)

    def action_logits(self, g, h_i, amask):
        return self.act(torch.cat([h_i, g], -1)).masked_fill(amask < 0.5, NEG)

    def partner_logits(self, h_i, H, pmask):
        return (torch.einsum("bd,bmd->bm", self.q(h_i), self.k(H)) / math.sqrt(self.d)).masked_fill(pmask < 0.5, NEG)

    def cont_logit(self, g, h_i, a_onehot):
        return self.cont(torch.cat([g, h_i, a_onehot], -1)).squeeze(-1)


@dataclass
class DecisionRecord:
    """Everything needed to recompute the log-probability of one decision point."""

    feats: np.ndarray
    mask: np.ndarray
    glob: np.ndarray
    valid: np.ndarray  # (B,) sub-decision n was taken
    choose_mask: np.ndarray  # (B, M)
    train: np.ndarray  # (B,)
    amask: np.ndarray  # (B, 4)
    action: np.ndarray  # (B,)
    pmask: np.ndarray  # (B, M)
    partner: np.ndarray  # (B,) -1 if none
    has_cont: np.ndarray  # (B,)
    cont: np.ndarray  # (B,)
    logp: float
    value: float


def act_at_point(net: DispatcherNet, de, budget: int, greedy: bool = False, gen: Optional[torch.Generator] = None) -> DecisionRecord:
    """Run the autoregressive sequence at one decision point of DispatchEnv ``de``; applies clearances."""
    w = de.window
    M = len(w.mask)
    feats = torch.as_tensor(w.feats)[None]
    mask = torch.as_tensor(w.mask)[None]
    glob = torch.as_tensor(w.glob)[None]
    rec = DecisionRecord(
        feats=w.feats, mask=w.mask, glob=w.glob, valid=np.zeros(budget, np.float32), choose_mask=np.zeros((budget, M), np.float32),
        train=np.zeros(budget, np.int64), amask=np.zeros((budget, N_ACTIONS), np.float32), action=np.zeros(budget, np.int64),
        pmask=np.zeros((budget, M), np.float32), partner=-np.ones(budget, np.int64), has_cont=np.zeros(budget, np.float32),
        cont=np.zeros(budget, np.float32), logp=0.0, value=0.0,
    )
    chosen = np.zeros(M, np.float32)
    logp = 0.0

    def pick(logits):
        if greedy:
            return int(torch.argmax(logits))
        return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))

    with torch.no_grad():
        for n in range(budget):
            cm = w.choosable * (1 - chosen)
            if cm.sum() == 0:
                break
            g, H = net.encode(feats, mask, glob, torch.as_tensor(chosen)[None], torch.tensor([n / budget]))
            if n == 0:
                rec.value = float(net.value(g))
            pl = net.pointer_logits(g, H, torch.as_tensor(cm)[None])[0]
            i = pick(pl)
            logp += float(F.log_softmax(pl, -1)[i])
            amask, pm = de.action_mask(i)
            al = net.action_logits(g, H[:, i], torch.as_tensor(amask)[None])[0]
            a = pick(al)
            logp += float(F.log_softmax(al, -1)[a])
            j = None
            if a == YIELD_TO:
                pml = net.partner_logits(H[:, i], H, torch.as_tensor(pm[YIELD_TO])[None])[0]
                j = pick(pml)
                logp += float(F.log_softmax(pml, -1)[j])
                rec.pmask[n] = pm[YIELD_TO]
                rec.partner[n] = j
            de.apply(i, a, j)
            chosen[i] = 1
            rec.valid[n], rec.choose_mask[n], rec.train[n], rec.amask[n], rec.action[n] = 1, cm, i, amask, a
            more = de.can_continue() and (w.choosable * (1 - chosen)).sum() > 0 and n + 1 < budget
            if not more:
                break
            cl = net.cont_logit(g, H[:, i], F.one_hot(torch.tensor([a]), N_ACTIONS).float())[0]
            p = torch.sigmoid(cl)
            c = int(p > 0.5) if greedy else int(torch.rand(1, generator=gen) < p)
            logp += float(F.logsigmoid(cl) if c else F.logsigmoid(-cl))
            rec.has_cont[n], rec.cont[n] = 1, c
            if not c:
                break
    rec.logp = logp
    return rec


def evaluate_records(net: DispatcherNet, batch: dict, budget: int):
    """Recompute joint log-prob, entropy and value for a batch of stacked DecisionRecords."""
    feats, mask, glob = batch["feats"], batch["mask"], batch["glob"]
    Bn, M = mask.shape
    logp = torch.zeros(Bn)
    ent = torch.zeros(Bn)
    value = None
    chosen = torch.zeros(Bn, M)
    for n in range(budget):
        valid = batch["valid"][:, n]
        g, H = net.encode(feats, mask, glob, chosen, torch.full((Bn,), n / budget))
        if n == 0:
            value = net.value(g).squeeze(-1)
        if valid.sum() == 0:
            break
        cm = batch["choose_mask"][:, n] + (1 - valid).unsqueeze(-1)  # keep rows finite where invalid
        pl = F.log_softmax(net.pointer_logits(g, H, cm), -1)
        i = batch["train"][:, n]
        logp += valid * pl.gather(1, i[:, None]).squeeze(1)
        ent += valid * -(pl.exp() * pl).sum(-1)
        h_i = H[torch.arange(Bn), i]
        am = batch["amask"][:, n] + (1 - valid).unsqueeze(-1)
        al = F.log_softmax(net.action_logits(g, h_i, am), -1)
        a = batch["action"][:, n]
        logp += valid * al.gather(1, a[:, None]).squeeze(1)
        ent += valid * -(al.exp() * al).sum(-1)
        isy = valid * (a == YIELD_TO).float()
        if isy.sum() > 0:
            pm = batch["pmask"][:, n] + (1 - isy).unsqueeze(-1)
            pll = F.log_softmax(net.partner_logits(h_i, H, pm), -1)
            j = batch["partner"][:, n].clamp(min=0)
            logp += isy * pll.gather(1, j[:, None]).squeeze(1)
        hc = batch["has_cont"][:, n]
        if hc.sum() > 0:
            cl = net.cont_logit(g, h_i, F.one_hot(a, N_ACTIONS).float())
            c = batch["cont"][:, n]
            logp += hc * (c * F.logsigmoid(cl) + (1 - c) * F.logsigmoid(-cl))
        chosen = chosen + valid.unsqueeze(-1) * F.one_hot(i, M).float()
    return logp, ent, value
