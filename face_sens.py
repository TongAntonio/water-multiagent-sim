"""Extended face-failure sensitivity (supplement to Table I / Table II).

Table II shows that the quality gate rejects most captures under strong low light / blur (73 % / 61 % in the degradation test).
Here the total face-route failure probability (fnmr + fta) is pushed far beyond the 1-30 % range of the main sensitivity
analysis, at rho = 0.2 on the test seeds 9000-9029.  Users whose face attempt fails fall back to their token (70 % hold one)
or to staff assistance (Section V pathways), exactly as in the main simulator.

    python face_sens.py run        # about 1-2 min on 2 vCPU -> results_facesens.csv
    python face_sens.py analyze    # -> out/facesens_summary.csv, out/facesens_contrasts.csv
"""
import sys, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
from run_eval import job, TEST

FNMR = 0.05                                  # kept at the default; the extra failure mass is placed on fta
TOTALS = [0.08, 0.30, 0.50, 0.75, 0.90]      # total face failure = fnmr + fta  (0.08 = default 0.05 + 0.03)
PFACE = [0.5, 1.0]                           # share of enrolled users on the face route (default 0.5; 1.0 = everyone)
CFGS = ["B0", "B1", "B3", "B4", "B5"]
MET = ["delivered_ratio", "wait_mean", "wait_p95", "wait_vuln", "pc", "unauth_rate", "assist_n", "vuln_missed_priority"]


def run():
    jobs = []
    for pf, tot in itertools.product(PFACE, TOTALS):
        for cfg in CFGS:
            for s in TEST:
                jobs.append(("facesens", cfg, dict(rho=0.2, p_face=pf, fnmr=FNMR, fta=round(tot - FNMR, 4)), s))
    print(len(jobs), "episodes", flush=True)
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=10)
    df = pd.DataFrame(res)
    df["total_fail"] = (df["p_fnmr"] + df["p_fta"]).round(3)
    df.to_csv("results_facesens.csv", index=False)
    print("done", len(df))


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2 or x.std(ddof=1) == 0:
        return m, m, m
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def analyze():
    df = pd.read_csv("results_facesens.csv")
    rows = []
    for (pf, tot, cfg), g in df.groupby(["p_p_face", "total_fail", "config"]):
        r = dict(p_face=pf, total_fail=tot, config=cfg, n=len(g))
        for m in MET:
            r[m] = g[m].mean()
        rows.append(r)
    pd.DataFrame(rows).to_csv("out/facesens_summary.csv", index=False)
    # paired contrasts (same seed)
    out = []
    for (pf, tot), g in df.groupby(["p_p_face", "total_fail"]):
        P = {c: g[g.config == c].set_index("seed") for c in CFGS}
        for a, b in [("B3", "B1"), ("B3", "B0"), ("B4", "B3"), ("B5", "B3")]:
            for m in ["delivered_ratio", "wait_vuln", "wait_p95", "pc", "unauth_rate"]:
                d = (P[a][m] - P[b][m]).dropna()
                mean, lo, hi = ci(d)
                out.append(dict(p_face=pf, total_fail=tot, contrast=f"{a}-{b}", metric=m, diff=mean, lo=lo, hi=hi, n=len(d)))
    pd.DataFrame(out).to_csv("out/facesens_contrasts.csv", index=False)
    # B3 vs its own default-failure twin (same seed, same request stream)
    base = df[(df.total_fail == 0.08)]
    own = []
    for (pf, tot), g in df.groupby(["p_p_face", "total_fail"]):
        for cfg in CFGS:
            a = g[g.config == cfg].set_index("seed")
            b = base[(base.p_p_face == pf) & (base.config == cfg)].set_index("seed")
            for m in ["delivered_ratio", "wait_vuln", "wait_p95", "pc", "assist_n"]:
                d = (a[m] - b[m]).dropna()
                mean, lo, hi = ci(d)
                own.append(dict(p_face=pf, total_fail=tot, config=cfg, metric=m, diff_vs_default_fail=mean, lo=lo, hi=hi))
    pd.DataFrame(own).to_csv("out/facesens_vs_default.csv", index=False)
    print("written")


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
