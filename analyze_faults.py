"""Aggregate results_faults.csv and results_breakdown.csv (seed = replication unit, 30 paired test seeds)."""
import numpy as np, pandas as pd
from scipy import stats

NS = 30
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan, np.nan, len(x))
    m, s = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, len(x) - 1) * s / np.sqrt(len(x))
    return (m, m - h, m + h, len(x))

# ------------------------------------------------------------------ faults
F = pd.read_csv("results_faults.csv")
base = F[F.fault == "none"].set_index(["config", "seed"])
rows = []
for (cfg, kind), g in F[F.fault != "none"].groupby(["config", "fault"]):
    rec = g.rec_min.values
    m_d, lo_d, hi_d, _ = ci(g.deficit_delivered * 100)
    m_w, lo_w, hi_w, _ = ci(g.d_wait_mean)
    m_a, lo_a, hi_a, _ = ci(g.d_abandon_rate * 100)
    rows.append(dict(config=cfg, fault=kind, n=len(g),
                     rec_median=np.median(rec), rec_q1=np.percentile(rec, 25), rec_q3=np.percentile(rec, 75),
                     rec_mean=rec.mean(), censored=int(g.censored.sum()),
                     no_degradation=int((g.rec_min == 0).sum()),
                     peak_mean=g.peak_excess.mean(), peak_sd=g.peak_excess.std(ddof=1),
                     area_mean=g.excess_area.mean(),
                     deficit_mean=m_d, deficit_lo=lo_d, deficit_hi=hi_d,
                     dwait_mean=m_w, dwait_lo=lo_w, dwait_hi=hi_w,
                     dabandon_mean=m_a, dabandon_lo=lo_a, dabandon_hi=hi_a,
                     dpc_mean=g.d_pc.mean(), dprotq=g.d_protq_viol.mean(),
                     dovr_mean=g.d_override.mean(),
                     max_down_deliv=g.deliv_while_down.max(), max_faulty_deliv=np.nanmax(g.deliv_from_faulty_L) if g.deliv_from_faulty_L.notna().any() else np.nan))
FR = pd.DataFrame(rows)
FR.to_csv("out/faults_summary.csv", index=False)

# paired contrasts between configurations within the same fault (seed-paired)
con = []
for kind in ["outage", "stockout", "partition"]:
    for a, b in [("B4", "B3"), ("B5", "B3"), ("B0", "B3")]:
        ga = F[(F.config == a) & (F.fault == kind)].set_index("seed")
        gb = F[(F.config == b) & (F.fault == kind)].set_index("seed")
        for met in ["rec_min", "deficit_delivered", "peak_excess", "d_wait_mean"]:
            d = (ga[met] - gb[met]).values * (100 if met == "deficit_delivered" else 1)
            m, lo, hi, _ = ci(d)
            con.append(dict(fault=kind, a=a, b=b, metric=met, diff=m, lo=lo, hi=hi,
                            dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan))
for kind in ["pol_crash", "pol_random"]:
    ga = F[(F.config == "B4") & (F.fault == kind)].set_index("seed")
    for met in ["rec_min", "deficit_delivered", "peak_excess", "d_wait_mean", "d_override"]:
        m, lo, hi, _ = ci(ga[met].values * (100 if met == "deficit_delivered" else 1))
        con.append(dict(fault=kind, a="B4", b="own fault-free twin", metric=met, diff=m, lo=lo, hi=hi, dz=np.nan))
pd.DataFrame(con).to_csv("out/faults_contrasts.csv", index=False)

# ------------------------------------------------------------------ breakdown
B = pd.read_csv("results_breakdown.csv")
def agg(kind, keys, cfgs, cols=("completed", "wait_mean", "wait_p95", "delivered_ratio", "delay_min", "prio_granted", "prio_full")):
    out = []
    for cfg in cfgs:
        for key in keys:
            g = B[(B.config == cfg) & (B.kind == kind) & (B.key == key)]
            r = dict(config=cfg, kind=kind, key=key, n_per_seed=g.n.mean(), n_total=int(g.n.sum()),
                     seeds_with_data=int((g.n > 0).sum()))
            for c in cols:
                m, lo, hi, k = ci(g[c].values)
                r[c] = m; r[c + "_lo"] = lo; r[c + "_hi"] = hi
            out.append(r)
    return pd.DataFrame(out)

cfgs = ["B0", "B1", "B3", "B4", "B5", "O1"]
G = agg("group", ["general", "elderly", "medical", "vulnerable"], cfgs)
R_ = agg("route", ["token", "face", "face_fb", "assist", "none", "misuse"], cfgs)
RV = agg("route_vuln", ["token", "face", "face_fb", "assist"], cfgs)
RS = agg("reason", ["face_fail_token", "not_enrolled", "token_lost", "face_fail_no_token", "all_assistance", "no_fallback_vuln"], cfgs)
for name, d in (("groups", G), ("routes", R_), ("routes_vuln", RV), ("reasons", RS)):
    d.to_csv(f"out/breakdown_{name}.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
print("=== FAULT SUMMARY ===")
print(FR[["config", "fault", "n", "rec_median", "rec_q1", "rec_q3", "censored", "no_degradation", "peak_mean", "deficit_mean", "deficit_lo", "deficit_hi", "dwait_mean", "dabandon_mean", "dovr_mean", "max_down_deliv", "max_faulty_deliv"]].round(2).to_string())
print("=== FAULT CONTRASTS ===")
print(pd.DataFrame(con).round(3).to_string())
print("=== GROUPS ===")
print(G[["config", "key", "n_per_seed", "completed", "wait_mean", "wait_p95", "delivered_ratio"]].round(3).to_string())
print("=== ROUTES (vulnerable) ===")
print(RV[["config", "key", "n_per_seed", "seeds_with_data", "completed", "wait_mean", "wait_p95", "delay_min", "prio_granted", "prio_full"]].round(3).to_string())
print("=== REASONS ===")
print(RS[["config", "key", "n_per_seed", "n_total", "seeds_with_data", "completed", "wait_mean", "wait_p95", "delay_min", "prio_granted", "prio_full", "delivered_ratio"]].round(3).to_string())
