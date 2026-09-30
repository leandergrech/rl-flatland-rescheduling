# References

Every source below was opened on 2026-09-30, either in this project's main session or by a
research sub-agent that fetched the page or PDF and returned the URL with each extracted fact.
Numbers quoted elsewhere in these docs link straight to the page they came from. Entries marked
**unverified** could not be opened in full, and nothing on this site depends on their numbers.

## Software environment (verified by installation)

Installed on 2026-09-30 into a fresh `.venv` (uv 0.10.11, CPython 3.12.3) on a ThinkPad with a
12th Gen Intel Core i7-1260P (16 logical CPUs), Linux 6.8.0.

| Package | Version | Licence | Used for |
|---|---|---|---|
| flatland-rl | 4.3.0 (released 2026-08-10) | MIT | the environment |
| torch | 2.14.1+cpu | BSD-3-Clause | PPO and imitation networks |
| numpy | 1.26.4 (flatland-rl pins `numpy<2`) | BSD-3-Clause | everything |
| networkx | 3.4.2 | BSD-3-Clause | dependency of flatland-rl |
| pandas | 2.3.3 | BSD-3-Clause | result tables |
| matplotlib | 3.10.9 | matplotlib licence (PSF-based) | plots |
| cython | 3.3.0 | Apache-2.0 | dependency of flatland-rl |
| gymnasium | 1.3.0 | MIT | installed with stable-baselines3, not imported by `rl_flatland` |
| stable-baselines3 | 2.9.0 | MIT | installed; not used, see [04-designs](04-designs.md#why-a-custom-ppo-trainer) |
| mkdocs | 1.6.1 | BSD-2-Clause | this site |
| mkdocs-material | 9.7.7 | MIT | this site |
| pytest | 9.1.1 | MIT | tests |

No third-party solver code is vendored. The OR baseline in `src/rl_flatland/baselines/or_planner.py`
was written from the published descriptions below. The 2020 winner's code
([Jiaoyang-Li/Flatland](https://github.com/Jiaoyang-Li/Flatland)) carries an academic/non-profit
licence that also forbids use in later Flatland challenges, so none of it was copied.

## Benchmark, competitions and official documentation

- <a id="laurent2021"></a>**Laurent et al. 2021.** F. Laurent, M. Schneider, C. Scheller, J. Watson, J. Li, Z. Chen, Y. Zheng, S.-H. Chan, K. Makhnev, O. Svidchenko, V. Egorov, D. Ivanov, A. Shpilman, E. Spirovska, O. Tanevski, A. Nikov, R. Grunder, D. Galevski, J. Mitrovski, G. Sartoretti, Z. Luo, M. Damani, N. Bhattacharya, S. Agarwal, A. Egli, E. Nygren, S. Mohanty. "Flatland Competition 2020: MAPF and MARL for Efficient Train Coordination on a Grid World." *PMLR* 133:275–301 (NeurIPS 2020 Competition and Demonstration Track), 2021. PDF: <http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf>; arXiv: <https://arxiv.org/abs/2103.16511>. Used for: 2020 leaderboard (Table 1, p.288), scoring and protocol (§2.3–2.4, pp.279–280), evaluation parameters (Appendix B, p.296), hardware (Appendix C, p.297), team methods (§4, pp.282–287), and the RL-versus-OR conclusion (§6, p.289).
- <a id="mohanty2020"></a>**Mohanty et al. 2020.** S. Mohanty, E. Nygren, F. Laurent, M. Schneider, C. Scheller, N. Bhattacharya, J. Watson, A. Egli, C. Eichenberger, C. Baumberger, G. Vienken, I. Sturm, G. Sartoretti, G. Spigler. "Flatland-RL: Multi-Agent Reinforcement Learning on Trains." arXiv:2012.05893, 2020. <https://arxiv.org/abs/2012.05893>. Used for: transition-map formalism (§2.2), observations (§2.4), reward formalism (§2.5), baseline table on 25×25 grids with 5 agents (Table 1, p.20).
- <a id="fl3-eval"></a>**Flatland 3 evaluation page.** Flatland book. <https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html>. Used for: normalised reward, time limits (10 min planning, 10 s per step, 2 h total, 4 CPU cores, 15 GB RAM).
- <a id="fl3-migration"></a>**Flatland 3 migration guide.** <https://flatland-association.github.io/flatland-book/challenges/flatland3/flatland-3-migration-guide.html>. Used for: what changed from Flatland 2 (line generators, off-map start, WAITING state).
- <a id="fl3-aicrowd"></a>**Flatland 3 challenge pages (AIcrowd).** Overview <https://www.aicrowd.com/challenges/flatland-3>, leaderboard <https://www.aicrowd.com/challenges/flatland-3/leaderboards>, winners <https://www.aicrowd.com/challenges/flatland-3/winners>. Used for: rounds and dates (Round 1 2021-09-17 to 2021-10-31 with equal speeds; Round 2 2021-11-01 to 2021-12-15 with variable speeds), 472 participants and 22 teams, final leaderboard.
- <a id="neurips2020-winners"></a>**NeurIPS 2020 Flatland winners post.** <https://discourse.aicrowd.com/t/neurips-2020-flatland-winners/4010>. Used for: top-3 of each track.
- <a id="book-agent"></a>**Flatland book, agent states.** <https://flatland-association.github.io/flatland-book/environment/environment/agent.html>; **actions** <https://flatland-association.github.io/flatland-book/environment/environment/actions.html>; **observations** <https://flatland-association.github.io/flatland-book/environment/observation_builder/provided_observations.html>; **rewards** <https://flatland-association.github.io/flatland-book/environment/environment/rewards.html>; **trajectories and policy evaluation** <https://flatland-association.github.io/flatland-book/environment/evaluation.html>; **Flatland Benchmarks** <https://flatland-association.github.io/flatland-book/challenges/flatland-benchmarks.html>.
- <a id="ecml2026"></a>**ECML 2026 Flatland challenge ("Real-World Baselines Challenge").** Overview <https://flatland-association.github.io/flatland-book/challenges/ecml2026.html>, evaluation <https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html>, levels <https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html>, results <https://flatland-association.github.io/flatland-book/challenges/ecml2026/post-competition_analysis.html>, starter kit <https://github.com/flatland-association/ecml2026-starterkit>. The competition paper, "Real-World Baselines Challenge: Dynamic Train Rescheduling under Stochastic Perturbations", is under review and was not available.
- <a id="flatland-rl"></a>**flatland-rl source.** <https://github.com/flatland-association/flatland-rl> (MIT), CHANGELOG <https://raw.githubusercontent.com/flatland-association/flatland-rl/main/CHANGELOG.md>, PyPI history <https://pypi.org/pypi/flatland-rl/json>. Version 4.3.0's source was read directly for the step function, MotionCheck, rewards and timetable generator. Note: PyPI dates 4.0.0 to 2023-10-27, while the GitHub release object for v4.0.0 is dated 2025-02-18.
- <a id="flatland-association"></a>**Flatland Association.** OpenRail Association news, 2024 (founded July 2023 in Bern, president Erik Nygren): <https://openrailassociation.org/news/2024/three-new-members-have-joined-openrail-association/>. Homepage <https://www.flatland-association.org/>.
- <a id="ai4realnet"></a>**AI4REALNET.** Horizon Europe grant 101119527, 6 M€, 42 months, Flatland plus Grid2Op plus BlueSky: <https://ai4realnet.eu/>; ZHAW project page (started 10/2023, 15 partners including SBB, DB InfraGO, TU Delft, Flatland Association): <https://www.zhaw.ch/en/research/project/74103>; GitHub organisation <https://github.com/AI4REALNET>.

## Operations-research entries

- <a id="andreica2021"></a>**Andreica 2021.** M.-I. Andreica. "Winning Solution of the AIcrowd SBB Flatland Challenge 2019-2020." arXiv:2111.07876. <https://arxiv.org/abs/2111.07876>. Interview: <https://www.aicrowd.com/blogs/flatland-mugurel>. Used for: time-expanded path planning, fast-trains-first priorities, keeping each cell's visiting order after malfunctions, 99% of agents routed. His code repo on gitlab.aicrowd.com could not be opened (**unverified** licence).
- <a id="li2021"></a>**Li et al. 2021.** J. Li, Z. Chen, Y. Zheng, S.-H. Chan, D. Harabor, P. J. Stuckey, H. Ma, S. Koenig. "Scalable Rail Planning and Replanning: Winning the 2020 Flatland Challenge." *Proc. ICAPS* 31:477–485, 2021. Page <https://ojs.aaai.org/index.php/ICAPS/article/view/15994>; PDF <https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487>. Code <https://github.com/Jiaoyang-Li/Flatland> (academic/non-profit licence). Used for: PP + SIPP + LNS + MCP + partial replanning + lazy planning, and the component ablations (Round 2: 362 instances, 98.5% success, score 297.507; largest fully solved instance 3,256 trains in 704 s).
- <a id="chen2023"></a>**Chen et al. 2023.** Z. Chen, J. Li, D. Harabor, P. J. Stuckey. "Scalable Rail Planning and Replanning with Soft Deadlines." arXiv:2306.06455. <https://arxiv.org/abs/2306.06455>. Used for: Flatland 3 winner (competition score 135.47, 145 of 150 instances; 140.99 with all 150 solved in a later rerun), slack-based priorities, delay-based LNS neighbourhoods, periodic partial replanning.
- <a id="ecml-winner"></a>**ECML 2026 winner "v4-sipp-locks".** <https://github.com/darshanmakwana412/ecml2026> (MIT). Prioritized SIPP by tightest slack, trains held off-map before departure, "overstay rights" for trains physically in a cell, directional locks on the 18 longest single-track corridors.
- <a id="ecml-runnerup"></a>**ECML 2026 runner-up "v5-dispatcher".** <https://github.com/Avinash837/ecml2026-starterkit/tree/v5-dispatcher-submission> (MIT). Fast-first reservation planner with cell and edge-interval reservations; trains that cannot be placed stay off-map.
- <a id="flatland-baselines"></a>**flatland-baselines deadlock-avoidance heuristic.** <https://github.com/flatland-association/flatland-baselines> (MIT), `flatland_baselines/deadlock_avoidance_heuristic/policy/deadlock_avoidance_policy.py`. A train follows its shortest path and moves only if enough free cells separate it from every opposing train on that path.

## Multi-agent path finding

- <a id="erdmann1987"></a>**Erdmann & Lozano-Pérez 1987.** "On Multiple Moving Objects." *Algorithmica* 2:477–521. Preprint MIT AI Memo AIM-883: <https://dspace.mit.edu/handle/1721.1/5602>.
- <a id="silver2005"></a>**Silver 2005.** "Cooperative Pathfinding." *Proc. AIIDE* 1:117–122. <https://ojs.aaai.org/index.php/AIIDE/article/view/18726>.
- <a id="sharon2015"></a>**Sharon, Stern, Felner & Sturtevant 2015.** "Conflict-based search for optimal multi-agent pathfinding." *Artificial Intelligence* 219:40–66. <https://digitalcommons.du.edu/computer_science_faculty/7/>.
- <a id="phillips2011"></a>**Phillips & Likhachev 2011.** "SIPP: Safe interval path planning for dynamic environments." *ICRA 2011*, 5628–5635, DOI 10.1109/ICRA.2011.5980306. Metadata <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ICRA.2011.5980306>. Only the bibliographic record was opened; the description of safe intervals used here also matches Li et al. 2021 §3.
- <a id="ma2019"></a>**Ma, Harabor, Stuckey, Li & Koenig 2019.** "Searching with Consistent Prioritization for Multi-Agent Path Finding" (PBS). *Proc. AAAI* 33:7643–7650. <https://ojs.aaai.org/index.php/AAAI/article/view/4758>. Code <https://github.com/Jiaoyang-Li/PBS> (USC research licence).
- <a id="li2021lns"></a>**Li, Chen, Harabor, Stuckey & Koenig 2021.** "Anytime Multi-Agent Path Finding via Large Neighborhood Search." *IJCAI 2021*, 4127–4135. Code and README <https://github.com/Jiaoyang-Li/MAPF-LNS>.
- <a id="li2022lns2"></a>**Li et al. 2022.** "MAPF-LNS2: Fast Repairing for Multi-Agent Path Finding via Large Neighborhood Search." *Proc. AAAI* 36(9). <https://ojs.aaai.org/index.php/AAAI/article/view/21266>.
- <a id="stern2019"></a>**Stern et al. 2019.** "Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks." *SoCS 2019*. <https://arxiv.org/abs/1906.08291>.
- <a id="atzmon2019"></a>**Atzmon, Diei & Rave 2019.** "Multi-Train Path Finding." *Proc. SoCS* 10(1):125–129. <https://ojs.aaai.org/index.php/SOCS/article/view/18515>.
- <a id="atzmon2020"></a>**Atzmon, Stern, Felner, Wagner, Barták & Zhou 2020.** "Robust Multi-Agent Path Finding and Executing." *JAIR*. <https://jair.org/index.php/jair/article/view/11734>.
- <a id="okumura2022"></a>**Okumura, Machida, Défago & Tamura 2022.** "Priority Inheritance with Backtracking for Iterative Multi-agent Path Finding" (PIBT). <https://arxiv.org/abs/1901.11282>.
- <a id="okumura2023"></a>**Okumura 2023.** "LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding." *AAAI 2023*. <https://arxiv.org/abs/2211.13432>.
- <a id="mapf-survey2025"></a>**"Where Paths Collide: A Comprehensive Survey of Classic and Learning-Based Multi-Agent Pathfinding."** arXiv:2505.19219, 2025. <https://arxiv.org/abs/2505.19219>.

## Reinforcement learning on Flatland

- <a id="roost2020"></a>**Roost et al. 2020.** D. Roost, R. Meier, S. Huschauer, E. Nygren, A. Egli, A. Weiler, T. Stadelmann. "Improving Sample Efficiency and Multi-Agent Communication in RL-based Train Rescheduling." arXiv:2004.13439. <https://arxiv.org/html/2004.13439>.
- <a id="jbr-repo"></a>**JBR_HSE code.** <https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution> (README is a stub; method from [Laurent et al. 2021](#laurent2021) §4.2).
- <a id="nips-baselines"></a>**NeurIPS 2020 Flatland baselines (RLlib).** <https://github.com/AIcrowd/neurips2020-flatland-baselines>. **DQN starter kit** announcement <https://discourse.aicrowd.com/t/updated-starter-kit-a-full-dqn-baseline-you-can-submit-out-of-the-box/3698>.
- <a id="kopacz2022"></a>**Kopacz, Mester, Kolumbán & Csató 2022.** "Standardized feature extraction from pairwise conflicts applied to the train rescheduling problem." SAMI 2022. <https://arxiv.org/abs/2204.03061>.
- <a id="egorov2022"></a>**Egorov & Shpilman 2022.** "Scalable Multi-Agent Model-Based Reinforcement Learning" (MAMBA). *AAMAS 2022*. <https://arxiv.org/abs/2205.15023>. Flatland results are only in a figure; exact percentages are **unverified**.
- <a id="jiang2023"></a>**Jiang, Zhang, Li, Chen & Zhu 2023.** "Multi-Agent Path Finding via Tree LSTM." AAAI 2023 MAPF workshop. <https://arxiv.org/abs/2210.12933>. Code <https://github.com/liqimai/flatland-marl>. Used for: Flatland 3 reward restatement (§2), the 15 official test stages and per-stage results (Table 6), the leaderboard comparison with 125.3 points (Table 7), curriculum and "5 days of training" for one phase.
- <a id="jaziri2024"></a>**Jaziri, Künzel & Ramesh 2024.** "Mitigating the Stability-Plasticity Dilemma in Adaptive Train Scheduling with Curriculum-Driven Continual DQN Expansion." CoLLAs 2025. <https://arxiv.org/abs/2408.09838>. Only the abstract was verified.
- <a id="castagna2026"></a>**Castagna et al. 2026.** A. Castagna, S. Zahlner, A. Egli, C. Eichenberger, D. Boos, M. Meyer, A. Fuxjäger. "Towards Autonomous Railway Operations: A Semi-Hierarchical Deep Reinforcement Learning Approach to the Vehicle Rescheduling Problem." arXiv:2605.10257, 2026-05-11. <https://arxiv.org/html/2605.10257v1>. Exact per-baseline percentages are only in a bar chart (Fig. 6) and are **unverified**.
- <a id="bourgeat2026"></a>**Bourgeat, Legrain & Cappart 2026.** "Imitation-Guided World Models for Multi-agent Train Rescheduling." *CPAIOR 2026*, LNCS 16595:82–100. DOI <https://doi.org/10.1007/978-3-032-27242-3_6>. Paywalled with no preprint: only the bibliographic record is verified. Method and numbers are **unverified**.
- <a id="adalat2026"></a>**Adalat, Hamel-De le Court & Belardinelli 2026.** "Contract-Based Compositional Shielding for Safe Multi-Agent Reinforcement Learning." EUMAS 2026. <https://arxiv.org/abs/2606.14130>. Uses Flatland as one of six environments; Flatland numbers **unverified**.
- <a id="ecml-rl-baseline"></a>**ECML 2026 organisers' RL baseline ("adaptive completion").** <https://github.com/dynamik1703/ecml2026-starterkit> (MIT, branch `rl-masked-baseline`): masked PPO/rerank policy with a deadlock-avoidance fallback on dense station clusters.
- <a id="alomb"></a>**alomb/FlatlandChallenge** (course project with D3QN, parameter-sharing PPO, curriculum). <https://github.com/alomb/FlatlandChallenge>. No numbers verified.

## Railway surveys and neighbouring benchmarks

- <a id="li2022survey"></a>**Li, Bai, Yao, Waller & Liu 2022.** "A Bibliometric Analysis and Review on Reinforcement Learning for Transportation Applications." arXiv:2210.14524. <https://arxiv.org/abs/2210.14524>. Reviews air traffic control (§4.2) and railway rescheduling in one framework.
- <a id="displib2025"></a>**Kloster, Luteberget, Mannino & Sartor 2025.** "DISPLIB: a library of train dispatching problems." arXiv:2509.12254. <https://arxiv.org/abs/2509.12254>.
- <a id="donatus2026"></a>**Donatus, Ter & Udekwe 2026.** "Multi-Agent Reinforcement Learning in Intelligent Transportation Systems: A Comprehensive Survey." arXiv:2508.20315v2. <https://arxiv.org/pdf/2508.20315>.

## Air traffic control and conflict resolution

- <a id="bluesky2016"></a>**Hoekstra & Ellerbroek 2016.** "BlueSky ATC Simulator Project: an Open Data and Open Source Approach." *ICRAT 2016*. <https://research.tudelft.nl/en/publications/bluesky-atc-simulator-project-an-open-data-and-open-source-approa/>. Code <https://github.com/TUDelft-CNS-ATM/bluesky> (MIT).
- <a id="brittain2019"></a>**Brittain & Wei 2019.** "Autonomous Air Traffic Controller: A Deep Multi-Agent Reinforcement Learning Approach." arXiv:1905.01303. <https://arxiv.org/abs/1905.01303>.
- <a id="brittain2021"></a>**Brittain, Yang & Wei 2021.** "A Deep Multi-Agent Reinforcement Learning Approach to Autonomous Separation Assurance." *J. Aerospace Information Systems* 18(12). <https://arxiv.org/abs/2003.08353>.
- <a id="brittain2022"></a>**Brittain & Wei 2022.** "Scalable Autonomous Separation Assurance with Heterogeneous Multi-Agent Reinforcement Learning." *IEEE T-ASE* 19. Listed at <https://wei.engineering.gwu.edu/papers/>.
- <a id="isufaj2022"></a>**Isufaj, Aranega Sebastia & Piera 2022.** "Toward Conflict Resolution with Deep Multi-Agent Reinforcement Learning." *J. Air Transportation* 30(3):71–80. <https://trid.trb.org/view/1953229>.
- <a id="ribeiro2022"></a>**Ribeiro, Ellerbroek & Hoekstra 2022.** "Improving Algorithm Conflict Resolution Manoeuvres with Reinforcement Learning." *Aerospace* 9(12):847. <https://api.openalex.org/works/doi:10.3390/aerospace9120847>. Exact deltas **unverified** (publisher page blocked).
- <a id="tada"></a>**TADA (Terminal Airspace Digital Assistant).** SESAR 3 JU exploratory research project 101166972, 2024-09 to 2027-02, University of Malta among the partners. <https://www.sesarju.eu/projects/TADA>.

## Unverified or not found

- Pham et al. 2019 (ATM Seminar), "A machine learning approach for conflict resolution in dense traffic scenarios with uncertainties": publisher pages blocked; not cited for numbers.
- The two ScienceDirect railway-AI reviews (S2590198226003301, S0968090X22001206): HTTP 403; not used.
- Official score of the ECML 2026 runner-up beyond the post-competition page (12.4925 is verified there; the repository itself states no score).
- JBR_HSE training hardware and wall-clock time: not reported anywhere we could open.
- Training compute for TreeLSTM (beyond "5 days" for one phase), Maze-Flatland and MAMBA: not reported in the opened text.
- Whether Laurent et al. Appendix B's 41-test protocol is exactly "Round 2" of 2020: consistent with the text, not stated.
