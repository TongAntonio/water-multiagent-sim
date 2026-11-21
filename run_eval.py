"""Final evaluation on TEST seeds 9000-9029 (never used for training, calibration or checkpoint selection).
All configurations of a given seed see the identical request stream, failures, replenishment and packet-loss draw."""
import sys, json, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from sim import *

TEST = list(range(9000, 9030))
THR = json.load(open("calibrated_thresholds.json"))
_M = {}

def model(name):
    if name not in _M:
        import torch
        from ppo import load
        _M[name] = load(name)
    return _M[name]

def make(cfg):
    from ppo import Learned
    thq, thv, thqj = THR["heur"]
    cq, cv, cj = THR["central"]
    table = {
        "B0": ("fcfs",  lambda: Heur(thq, thv, thqj)),
        "B1": ("claim", lambda: Heur(thq, thv, thqj)),
        "B2": ("claim", lambda: Learned(model("B2_bc"))),
        "B3": ("auth",  lambda: Heur(thq, thv, thqj)),
        "B4": ("auth",  lambda: Learned(model("B4_bc"))),
        "B5": ("auth",  lambda: Central(cq, cv, cj)),
        "O1": ("oracle", lambda: Learned(model("B4_bc"))),
        # supplementary
        "B3-NR": ("auth", lambda: Heur(use_ref=False)),
        "B0-NR": ("fcfs", lambda: Heur(use_ref=False)),
        "B4-scratch": ("auth", lambda: Learned(model("B4_scratch"))),
    }
    return table[cfg]

def job(a):
    tag, cfg, pkw, seed = a
    regime, mk = make(cfg)
    P = Params(**pkw)
    m = run_one(seed, P, regime, mk())
    m.update(tag=tag, config=cfg, **{f"p_{k}": v for k, v in pkw.items()})
    return m

def main():
    t0 = time.time()
    core = ["B0", "B1", "B2", "B3", "B4", "B5", "O1"]
    supp = ["B3-NR", "B0-NR", "B4-scratch"]
    jobs = []
    # S1: main (iid packet loss)
    for rho in (0.0, 0.2, 0.4):
        for cfg in core + supp:
            for s in TEST:
                jobs.append(("main", cfg, dict(rho=rho), s))
    # S2: burst loss and a 2 h partition (long-run loss rate 0.2)
    for mode in ("burst", "partition"):
        for cfg in core:
            for s in TEST:
                jobs.append((mode, cfg, dict(rho=0.2, loss_mode=mode), s))
    # S3: scalability (policy is parameter-shared, so it is applied unchanged)
    for n in (10, 40):
        for cfg in core:
            for s in TEST:
                jobs.append((f"scale{n}", cfg, dict(rho=0.2, n=n), s))
    # S4: sensitivity of identity-related assumptions (rho = 0.2)
    sens = {"p_false": [0.0, 0.04, 0.16], "c_enr": [0.4, 0.6, 1.0], "fnmr": [0.01, 0.15, 0.30],
            "p_face": [0.0, 1.0], "p_lost": [0.20, 0.40], "p_acc": [0.4, 1.0]}
    for par, vals in sens.items():
        for v in vals:
            for cfg in ["B1", "B2", "B3", "B4", "B5"]:
                for s in TEST:
                    jobs.append((f"sens_{par}", cfg, {"rho": 0.2, par: v}, s))
    print(len(jobs), "episodes", flush=True)
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=20)
    df = pd.DataFrame(res)
    df.to_csv("results_raw.csv", index=False)
    print("done", len(df), "rows", round(time.time() - t0), "s")

if __name__ == "__main__":
    main()
