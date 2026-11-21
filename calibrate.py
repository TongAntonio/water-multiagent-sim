"""Calibration on VALIDATION seeds only (5000-5005): thresholds of the rule-based referral used by B0/B1/B3
(and the centralised B5).  Arrival intensity was fixed a priori by the stress target
'mean delivered ratio of B0 at rho=0.2 is about 0.85' (base_rate=0.18)."""
import itertools, json, sys
import numpy as np
from multiprocessing import Pool
from sim import *

VAL = list(range(5000, 5006))
BASE = 0.18

def job(args):
    kind, thq, thv, thqj, rho, seed = args
    P = Params(base_rate=BASE, rho=rho)
    if kind == "heur":
        ctl = Heur(thq, thv, thqj, use_ref=thq < 999)
    else:
        ctl = Central(thq, thv, thqj)
    m = run_one(seed, P, "auth", ctl)
    return (kind, thq, thv, thqj, rho, m["delivered_ratio"], m["wait_p95"])

if __name__ == "__main__":
    grid = [(999, 40.0, 1)] + [(a, b, c) for a in (3, 4, 6, 8, 12) for b in (20.0, 40.0, 60.0) for c in (0, 1, 2, 3)]
    jobs = [(kind, *g, rho, s) for kind in ("heur", "central") for g in grid for rho in (0.0, 0.2, 0.4) for s in VAL]
    jobs = [j for j in jobs if not (j[0] == "central" and j[1] == 999)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=8)
    import collections
    agg = collections.defaultdict(list)
    for kind, a, b, c, rho, dr, w in res:
        agg[(kind, a, b, c)].append((dr, w))
    out = {}
    for kind in ("heur", "central"):
        rows = sorted(((np.mean([x[0] for x in v]), -np.mean([x[1] for x in v]), k) for k, v in agg.items() if k[0] == kind), reverse=True)
        print(kind, "top5:")
        for r in rows[:5]:
            print("  dr=%.4f p95=%.1f %s" % (r[0], -r[1], r[2][1:]))
        out[kind] = rows[0][2][1:]
        norf = [r for r in rows if r[2][1] == 999]
        if norf:
            print("  no-referral dr=%.4f" % norf[0][0])
    json.dump({k: list(v) for k, v in out.items()}, open("calibrated_thresholds.json", "w"))
    print(out)
