---
icon: fl/learning
---

# :fl-learning: Learning on top of a planner: related work

!!! abstract "In short"

    This page asks whether anyone has already built what [TADA on rails](../08-tada-dispatcher.md)
    builds: a learned dispatcher that leaves a conflict-free plan in charge and may only edit it
    through checked clearances (hold, yield to a partner, reroute).

    - **One close precedent on Flatland.** A 2022 master's thesis at TU Darmstadt put a learned
      policy on top of a conflict-free timetable. After each delay, the policy picks one of ten
      wait-or-reroute resolutions per conflict, with infeasible ones masked
      ([Leichthammer 2022](../07-references.md#leichthammer2022)). It found the same thing this project found:
      simple baselines did better than every trained network.
    - **The parts exist separately elsewhere.**
        - Optimisation methods re-order a multi-robot plan online after delays and stay
          deadlock-free ([Berndt et al.](../07-references.md#berndt2023); [Feng et al. 2024](../07-references.md#feng2024)).
        - Learned priority orders improve prioritized planning in warehouses
          ([Zhang et al. 2022](../07-references.md#zhang2022); [Zheng et al. 2026](../07-references.md#zheng2026)).
        - On real rail lines, reinforcement learning has picked train orders while an optimiser
          keeps the timetable feasible ([Zhang et al. 2024](../07-references.md#zhang2024)). There the learned
          method also stayed behind the optimiser.
        - Safe-RL calls a feasibility mask like ours a *preemptive shield*
          ([Alshiekh et al. 2018](../07-references.md#alshiekh2018)).
    - **What this search did not find:** a learned policy that commits each edit only after
      re-planning it against a live reservation table, so that the plan stays deadlock-free by
      construction. That combination appears to be new to this project. Its result, no
      measurable gain over the planner, repeats the 2022 thesis.

    This is a search, not a proof of absence. It covered four strands and opened every source
    cited below on 2026-10-02 ([how it was done](#how-this-search-was-done)).

## The closest work, side by side

| Work | Domain | What is learned | What keeps it safe | Edits a running plan? | Against the classical baseline |
|---|---|---|---|---|---|
| **This project** ([TADA on rails](../08-tada-dispatcher.md)) | Flatland, 30 to 100 trains | Which train in a window of ≤ 8 gets HOLD, YIELD_TO or REROUTE (PPO) | Each edit re-planned with SIPP against a live, re-timed reservation table; committed only if feasible | Yes, online, after delays | Ties the planner (net +1 train of 300 on medium with malfunctions) |
| [Leichthammer 2022](../07-references.md#leichthammer2022) | Flatland (a fork), 20 to 50 trains | Which of 10 wait/reroute resolutions to apply to a detected conflict (Dueling Double DQN) | Infeasible resolutions masked; follow-up conflicts queued; no deadlock guarantee | Yes, online, after delays | All networks below the waiting and heuristic baselines |
| [Zhang et al. 2024](../07-references.md#zhang2024) | Real line (Utrecht–'s-Hertogenbosch) | The independent train-order variables of a MILP (double DQN) | An LP retimes the trains for the chosen orders | Re-solves at each step | 75.08% delay reduction against 81.95% for the optimiser |
| [Berndt et al.](../07-references.md#berndt2023) | Warehouse robots (AGVs) | Nothing: a MILP | Switchable dependency graph kept acyclic, so recursively feasible and deadlock-free | Yes, online re-ordering after delays | Up to 25% lower overall route completion time than the baselines it compares with |
| [Zheng et al. 2026](../07-references.md#zheng2026) | Warehouse lifelong MAPF | Priority orders for a rolling-horizon prioritized planner (PPO, attention) | Planner plus a repair step: conflict-free within the window, but no deadlock guarantee | Re-plans each window; delays not modelled | About 25% higher throughput than random priorities |
| [Bourgeat et al. 2026](../07-references.md#bourgeat2026) | Flatland, 5 to 15 trains | A policy imitating the 2020 winner's planner, then RL in a world model | None (deadlocks counted, not prevented) | No: replaces the planner at run time | 80.8% arrived at 10 trains against 98.0% for the planner it imitates |

## Strand 1: Flatland

**Leichthammer (2022)** is the closest precedent. In this TU Darmstadt master's thesis, trains
follow a conflict-free timetable. When a delay creates a conflict, a controller offers ten
resolutions: four retimings (wait here, or as close as possible to the conflict, for either train)
and six reroutes (the three shortest paths for each of the two trains). A Dueling Double DQN picks
one, and invalid options are masked at evaluation. The thesis evaluates five settings, from a
double track to random 60×60 maps with 50 trains, and concludes that "the waiting and heuristic
baseline solutions performed better than all deep networks" (§6.4.1, Table 6.2). On random maps
with 20, 35 and 50 trains, the waiting baseline completed 0.917, 0.892 and 0.831 of the journeys.
The best network on each size reached 0.864, 0.875 and 0.822.

It differs from this project in four ways:

- It checks feasibility locally (the conflict and its first follow-ups), not by re-planning
  against the whole reservation table.
- It is not deadlock-free by construction: the thesis reports that its heuristic baseline
  sometimes chose resolutions that led to deadlock.
- It resolves one conflict at a time instead of choosing among a window of trains.
- It uses DQN, not PPO.

The direction and the null result are the same.

**The 2020 competition** had hybrids of a different kind ([Laurent et al. 2021, §4](../07-references.md#laurent2021)):

- JBR_HSE gated departures with a learned classifier.
- Netcetera assigned priorities from a conflict graph.
- MARMot-Lab-NUS hard-coded masks and a one-train-per-cluster rule.

None of these ran a learned layer over a timed conflict-free plan.

**Recent work replaces the planner rather than editing its plan.**

- [Bourgeat et al. (2026)](../07-references.md#bourgeat2026) imitate the 2020 winner's planner (PP + SIPP + LNS +
  MCP), queried repeatedly during training, then fine-tune in a world model. Trained at 10 trains,
  the policy delivers 98.2, 80.8 and 71.3% of trains at 5, 10 and 15 trains. The planner it
  imitates delivers 100.0, 98.0 and 98.1% (Table 1).
- [Castagna et al. (2026)](../07-references.md#castagna2026) learn separate dispatch and routing
  policies from MCTS, with action masks but no conflict-free plan underneath.

**Repairing a plan without learning.** [Nygren et al. (2023)](../07-references.md#nygren2023) keep a
conflict-free plan and re-solve only a "scope" of trains around a malfunction with an
answer-set solver. They propose learning the scope but leave it to future work.

## Strand 2: re-ordering a plan after delays (multi-robot path finding)

Executing a plan in its planned order is the action dependency graph of
[Hönig et al. (2019)](../07-references.md#honig2019), the same principle as the MCP used here. Several lines of work
then *change* that order after delays while keeping execution deadlock-free. These are the
optimisation counterparts of a YIELD_TO clearance:

- **Switchable action dependency graph.** [Berndt et al.](../07-references.md#berndt2023) make the passing-order
  dependencies switchable and re-optimise them with a MILP in a receding horizon, in a
  "recursively feasible manner". They report up to 25% lower overall route completion time.
- **Switchable temporal plan graph.** [Feng et al. (2024)](../07-references.md#feng2024) search for the best passing
  orders after a delay, keeping the paths fixed, in under a second on small and medium maps.
  [Su et al. (2024)](../07-references.md#su2024) pre-verify pairs whose order can be swapped either way and decide
  them first-come-first-served at run time.
- **Learning in these frameworks** so far decides *when* to replan
  ([Zahrádka et al. 2026](../07-references.md#zahradka2026): a network over 42 dependency-graph features recovers up
  to 94.6% of the achievable reduction in delay impact). It also predicts execution times to
  rank order reversals before execution ([Yan et al. 2026](../07-references.md#yan2026), up to 40% normalised
  improvement).

None of these learns which order to switch online. A non-learned YIELD_TO search in the style of
switchable-edge search is the baseline this project's dispatcher did not have.

## Strand 3: learning priorities and neighbourhoods for a planner

Learning *which agents go first* before planning has worked in multi-robot path finding.

- [Zhang et al. (2022)](../07-references.md#zhang2022) learn a ranking over 26 per-agent features that imitates the
  best of many prioritized-planning runs.
- [Zheng et al. (2026)](../07-references.md#zheng2026) train PPO with an attention encoder to emit priority orders
  for a rolling-horizon prioritized planner in warehouses. They report about 25% higher
  throughput than random priorities. Their repair step keeps each window conflict-free but, in
  their words, "does not guarantee progress, completeness, or deadlock resolution".
- Learning also guides which agents to replan in large neighbourhood search
  ([Huang et al. 2022](../07-references.md#huang2022)), and a MARL policy can replan small neighbourhoods inside it
  ([Wang et al. 2025](../07-references.md#wang2025)).

All of this works before execution, without delays, and outside rail. Learned priorities for
Flatland with malfunctions remain untested (open question 3).

## Strand 4: railway dispatching with learning and optimisation

On real-rail models, the work closest to this project lets learning choose ordering decisions
while an optimiser or simulator keeps the result feasible.

- **[Zhang et al. (2024)](../07-references.md#zhang2024).** A double DQN sets the independent train-order binaries of
  a MILP; an LP retimes the trains. On a Dutch line it cut delay by 75.08% against FIFO, where
  the full optimiser cut 81.95%.
- **[Liu et al. (2026)](../07-references.md#liu2026).** Trained networks predict order and composition decisions,
  with a heuristic fallback that guarantees feasibility.
- **[Ghasempour et al.](../07-references.md#ghasempour2019).** Approximate dynamic programming sequences trains at
  junctions over the sequences a microscopic simulator confirms as realisable. Its baseline is
  first-come-first-served.
- **Without an optimiser.** RL dispatchers either mask infeasible moves without a deadlock
  guarantee ([Agasucci et al. 2023](../07-references.md#agasucci2023), 4 to 5 deadlocks per 100 instances), or
  pre-filter moves with a look-ahead rule that the authors say does not guarantee deadlock
  avoidance ([Prasad et al. 2020](../07-references.md#prasad2020)).
- **Surveys.** Two of them name the combination of mathematical programming and machine learning
  as a direction rather than an established practice ([Tang et al. 2022, §4](../07-references.md#tang2022);
  [Bešinović et al. 2022, §VI.B](../07-references.md#besinovic2022)).

## Strand 5: air traffic control and shielded RL

- **Air traffic.** At TU Delft, [Ribeiro et al. (2022)](../07-references.md#ribeiro2022) let an RL
  agent choose the look-ahead time and manoeuvre types of a geometric resolver (MVP) and reduced
  losses of separation, with no feasibility check. No public TADA publication describes the
  windowed dispatcher whose structure this project borrows; it comes from the author's own
  project (see [the analogy](../08-tada-dispatcher.md#the-analogy)).
- **Shielding.** Safe-RL calls a check that removes unsafe actions before the agent chooses a
  *preemptive shield* ([Alshiekh et al. 2018](../07-references.md#alshiekh2018)). Factored shields kept multi-agent
  learners collision-free with comparable reward ([ElSayed-Aly et al. 2021](../07-references.md#elsayed2021)).
  Shields are usually synthesised offline from a formal specification. Here the planner computes
  the mask online, for each clearance. Contract-based shielding has been shown on a two-train
  Flatland instance with yield obligations
  ([Adalat et al. 2026](../07-references.md#adalat2026)).
- **Editing a complete schedule by feasible moves** also appears in job-shop scheduling, where
  RL picks swaps in an existing schedule ([Zhang et al. 2024, ICLR](../07-references.md#zhang2024jsp)). It runs
  offline, as an improvement heuristic.

## What this means for the experiment

1. **The idea is not unprecedented on Flatland, and neither is the result.** Leichthammer (2022)
   and this project both put learning on top of a conflict-free plan, and both found that
   learning did not beat simple baselines. Zhang et al. (2024) report the same on a real line
   against an optimiser.
2. **What is new here is the safety mechanism, not the learning.** Every committed edit is
   re-planned against the live reservation table, and no evaluation, including 200 episodes of
   random edits, ended in a deadlock. Within this search, no other learned dispatcher has that
   property.
3. **The missing comparison is optimisation, not more learning.** Online re-ordering after
   delays is solved by search in multi-robot execution (switchable dependency graphs and temporal
   plan graphs). Running that search over the same clearances would show how much there is to
   gain at all, before asking whether a policy can learn it.
4. **Where learning has helped is upstream.** The positive results learn priorities or
   neighbourhoods for a planner before execution, which is open question 3. Learning to repair
   after delays (open question 1) has optimisation baselines but, in this search, no learned
   success yet ([Open questions](../06-open-questions.md)).

## How this search was done

Four searches ran in parallel on 2026-10-02:

1. Flatland-specific work.
2. Multi-robot path finding with learning or online re-ordering.
3. Railway dispatching with learning and optimisation.
4. Air traffic control and shielded or residual RL.

Each search opened every source it reports. The closest works were then re-opened and their
numbers checked against the PDFs: Leichthammer, Zhang et al. 2024, Bourgeat et al., Berndt et
al., Zheng et al., Zahrádka et al. and Yan et al.

**Not verified:**

- Full texts behind publisher blocks: Šemrov et al. 2016, Khadilkar 2019, the Gatekeeper paper
  (IJCNN 2022), and Dalle & Parmentier 2022.
- An ML-guided configuration of RECIFE-MILP (RailDresden 2025), seen only as a search snippet.
- Any public TADA deliverable.

None of the conclusions above depends on these. Entries are listed under
[References](../07-references.md#learning-combined-with-planning-or-optimisation), with the
anchors used on this page.
