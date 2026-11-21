# Multi-agent water-distribution simulator and identity evaluation

Code, data and results behind the paper

> *An Integrated Security-Aware Multi-Agent Framework for Priority-Aware, Geo-Tracked, and Identity-Verified Water Distribution in Smart-City Disaster Relief* (IEEE conference draft).
> Citation / DOI: to be added when the paper is published.

Contents: a discrete-event multi-agent simulator (configurations B0-B5, O1), parameter-sharing PPO, fault-injection and failure-recovery experiments,
a face-route sensitivity analysis, and the identity-evaluation script for Table II (LFW verification / open-set identification with MobileFaceNet).
Every number in the results tables of the paper is produced by the scripts here; the raw per-episode outputs (`results_*.csv`) are included.

## Authorship and AI assistance

The research questions and the configurations to be compared (Tables I-III of the paper) are the authors'. The simulator, training, analysis and identity-evaluation code in this repository
was written by **Claude (Anthropic)** under the authors' direction, including several modelling choices that the paper does not specify (listed in *Modelling choices that are mine* below).
The authors ran the identity evaluation (Table II) on their own computer.
This is disclosed in the paper's Acknowledgment section in line with the IEEE policy on AI-generated content.

## Licences of third-party material (not included in this repository)

* **LFW** (Labeled Faces in the Wild) images and pairs are downloaded by the script; follow the LFW terms of use. LFW contains adult public figures only.
* **Face models** (`buffalo_s` / `buffalo_l`, SCRFD + MobileFaceNet / ArcFace-R50, insightface project) are downloaded by the script and are for **research / non-commercial use only**.
* **Code licence:** none chosen yet (all rights reserved until the repository owner adds a `LICENSE` file).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate        # Windows: py -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
python identity_eval.py --selftest                         # 14 unit tests, no data needed
python test_plumbing.py                                    # plumbing test of the Table II script with synthetic embeddings
```

Measured Table II outputs (MobileFaceNet, MacBook Pro with a 2.3 GHz 8-core Intel Core i9, macOS 26.7.1, 1 thread) are in `table2_lfw_mobilefacenet/`.
The extended face-failure sensitivity is `python face_sens.py run && python face_sens.py analyze` (about 1 minute).

---

Paper: *An Integrated Security-Aware Multi-Agent Framework for Priority-Aware, Geo-Tracked, and Identity-Verified Water Distribution in Smart-City Disaster Relief*

## What is here

| File | Purpose |
|---|---|
| `sim.py` | Discrete-event multi-agent simulator, queue manager (Eq. 13-14), safety supervisor, identity pathways, lossy links, metrics (Table III) and the deterministic controllers (B0, B1, B3, B5) |
| `ppo.py` | Parameter-sharing PPO, centralised critic, behaviour-cloning warm start (B2, B4, O1) |
| `calibrate.py` | Referral-threshold calibration on validation seeds only |
| `run_eval.py` | Final evaluation: 3,990 episodes on test seeds 9000-9029 |
| `analyze.py` | Statistics (paired differences, 95% CI, d_z, Holm), tables, figures |
| `faults.py` | Failure-recovery experiments (outage, stock-out, partition, policy crash, random policy proposals; 540 episodes) and the per-group / per-route / per-fallback-reason breakdown (180 episodes) -> `results_faults.csv`, `results_breakdown.csv` |
| `analyze_faults.py` | Aggregation of the two files above (seed = replication unit, t-based 95% CI) -> `out/faults_*.csv`, `out/breakdown_*.csv` |
| `gen_tables.py` | Writes the LaTeX tables of Section IX of the manuscript to `out/paper/` |
| `sim_v1_backup.py` | Simulator before the fault hooks were added; `faults.py` changes reproduce its results exactly (0 mismatches on 15 episodes x 12 metrics) |
| `face_sens.py` | Face-failure sensitivity up to 90 % (B0, B1, B3, B4, B5; 1,500 episodes) -> `results_facesens.csv`, `out/facesens_*.csv` |
| `identity_eval.py` | **Table II** script (LFW verification + open-set identification + acquisition failures + latency/memory/energy) |
| `test_plumbing.py` | Plumbing test of `identity_eval.py` with synthetic embeddings |
| `results_raw.csv` | One row per episode (all metrics) |
| `out/` | Tables (`.md`, `.tex`, `.csv`) and figures |
| `models/` | PPO checkpoints (`B2_bc.pt`, `B4_bc.pt`, `B4_scratch.pt`) |

## Reproduce Table I (about 3 minutes on 2 vCPU)

```bash
pip install numpy scipy pandas matplotlib torch
python calibrate.py                       # validation seeds 5000-5005 -> calibrated_thresholds.json
python ppo.py B4 bc 120 16 ; python ppo.py B2 bc 120 16 ; python ppo.py B4 scratch 120 16   # about 11 min
python run_eval.py                        # test seeds 9000-9029 -> results_raw.csv
python analyze.py                         # -> out/
```

Seeds are disjoint: train 1000-4999, heuristic calibration 5000-5005, PPO checkpoint selection 6000-6003, test 9000-9029.

## Run Table II on your own computer (the part that could not be run in the cloud)

The cloud session could not reach the LFW hosts, so Table II was not measured there.

```bash
pip install numpy scipy opencv-python onnxruntime
python identity_eval.py --selftest                              # 14 unit tests of the metric code
python identity_eval.py --download --model mbf --out t2_mbf     # downloads LFW (233 MB) + models (research licence), runs everything
python identity_eval.py --download --model r50 --out t2_r50 --threads 1
```

Works on Windows, macOS and Linux (Python 3.9+; on Windows peak memory is read from `PeakWorkingSetSize`; RAPL energy is Linux/Intel only, otherwise pass a power-meter reading).
If `python` is not found on Windows, use `py`.
If `vis-www.cs.umass.edu` does not resolve, `--download` falls back to the figshare mirror (SHA-256 verified); if both fail, download `lfw-funneled.tgz` and `pairs.txt` in a browser into `lfw_data/` and re-run.

Outputs `t2_*.json`, `t2_*.md`, `t2_*.tex` (rows of Table II). Expect roughly 10-25 minutes per model on a laptop.
Energy per attempt: run a USB power meter during the benchmark loop and pass `--measured-energy-j J --measured-attempts N`
(on Linux/Intel the script uses RAPL when readable; RAPL includes idle power).
For the Raspberry Pi / Jetson rows, run the same command on that device.

Protocol (all thresholds are chosen on validation data only):
* Verification: LFW 10-fold pairs, tau_ver chosen on the 9 other folds, applied to the held-out fold; Clopper-Pearson CI.
* Identification: open-set, Eq. (9)-(10), identity-disjoint calibration/evaluation split, gallery sizes 2/10/50/100/500, FPIR/FNIR.
* Only the detector and the embedding network are loaded. No age/gender/health model is used.

## Modelling choices that are mine (not specified in the paper) - please review

1. **B1 vs B3**: B1/B2 use *unverified claims* (90% of vulnerable users claim their category, 8% of general users claim falsely); B3/B4/B5 use authenticated eligibility through the credential / face / assistance pathways of Section V. B2 = learned coordination with unverified claims; O1 = B4 policy with oracle identity.
2. **Priority compliance PC** is evaluated against the *intended* ordering computed from ground-truth eligibility. Every config serves its own queue-manager candidate, so PC measures identity/claim errors and FCFS, not the learned controller choosing a wrong request.
3. **Identity error rates (FNMR 5%, FTA 3%, FMR 1e-3, credential loss 5%, enrolment 80%, staff assistance delay 10 min, copied-credential misuse 0.5%)** are explicit *sensitivity parameters*, not measurements (the paper allows this in Section VIII-C). Replace them with Table II measurements when available.
4. **Learned policies**: PPO from scratch did **not** discover referral within 120 updates x 16 episodes and converged to "always serve" (identical to B3 without referral: `B4-scratch` row). B2/B4 therefore use a **behaviour-cloning warm start** from the calibrated rule-based referral policy (99% imitation accuracy) followed by PPO fine-tuning; the checkpoint with the best validation return is used. The team reward of Eq. (16) is decomposed additively over dispensers; each agent is trained on a neighbourhood-averaged share of it.
5. Rule-based referral thresholds (queue >= 12, remaining volume >= 60 L, neighbour queue <= 2) were calibrated on validation seeds, so B0/B1/B3 are strong baselines (referral alone adds 3.6 points of delivered volume: B3 vs B3-NR).
6. Geographic and recent-service terms of Eq. (13) are disabled (weights renormalised), as the paper requires an operational definition before use.
7. One packet-loss draw is shared by every configuration of a seed; no retransmission is configured, so application-level delivery rate = 1 - rho.

## Limits

All workload, failure, acceptance and identity parameters are assumptions; absolute values have no empirical meaning, only the paired comparisons do.
Latency numbers produced in the cloud session (2 vCPU) are a pipeline check, not target-hardware results.

## Failure recovery and breakdowns (Section IX-C and IX-E of the manuscript)

```bash
python faults.py all          # about 1 minute on 2 vCPU; test seeds 9000-9029, rho = 0.2
python analyze_faults.py      # -> out/faults_summary.csv, out/faults_contrasts.csv, out/breakdown_*.csv
python gen_tables.py          # -> out/paper/*.tex
```

Faults are drawn from a separate generator (seed + 7,000,003), so every faulted episode differs from its fault-free twin only through the fault.
Window 60 min, start uniform in [0.25 H, 0.50 H); outage and stock-out affect 4 of 20 dispensers.
Recovery time = time after the fault ends until the backlog excess over the twin stays <= max(2, 10 % of the mean backlog) for 5 consecutive intervals; runs that never recover are right-censored.
Safety invariant checked in all 420 faulted episodes: no water is delivered by a dispenser that is down or inside a stock-out window.
These analyses are exploratory (95 % CI only, no multiplicity correction).

Modelling choices of the fault experiments that are mine: the stock-out empties the tank and blocks deliveries for the window and one 1,000 L emergency resupply arrives at its end;
the partition cuts the network into a west and an east half (the central node of B5 stays with the west half);
"policy crash" replaces the learned policy by the local rule (priority + aging, no referral) as in Section VI-D; "random proposals" is a stress test, not a failure mode specified in the paper.
