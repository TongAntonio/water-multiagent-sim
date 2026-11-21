"""Generate the LaTeX tables (IEEEtran, booktabs) that are inserted into the manuscript. Output: out/paper/*.tex"""
import os
import numpy as np, pandas as pd

os.makedirs("out/paper", exist_ok=True)
raw = pd.read_csv("results_raw.csv")
con = pd.read_csv("out/contrasts_primary.csv")
fsum = pd.read_csv("out/faults_summary.csv")
fcon = pd.read_csv("out/faults_contrasts.csv")
G = pd.read_csv("out/breakdown_groups.csv"); RV = pd.read_csv("out/breakdown_routes_vuln.csv"); RS = pd.read_csv("out/breakdown_reasons.csv")
B = pd.read_csv("results_breakdown.csv")


def n(x, nd=2, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{abs(x):.{nd}f}"
    neg = x < 0 and float(s) != 0.0
    sign = "$-$" if neg else ("$+$" if plus else "")
    return f"{sign}{s}"


def pm(m, s, nd):
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}"


def write(name, txt):
    open(f"out/paper/{name}.tex", "w").write(txt)


# ---------------------------------------------------------------- main table
LAB = {"B0": "B0", "B1": "B1", "B2": "B2", "B3": "B3", "B4": "B4", "B5": "B5", "O1": "O1",
       "B0-NR": "B0-NR", "B3-NR": "B3-NR", "B4-scratch": "B4-scratch"}
main = raw[raw.tag == "main"]
L = [r"\begin{table*}[!t]", r"\caption{Service and priority outcomes (mean $\pm$ SD over 30 paired test seeds; vulnerable = elderly or medical requests; PC = pairwise priority compliance, Eq.~\eqref{eqn19}; PC of O1 is 1 by construction). B0-NR, B3-NR and B4-scratch exchange no referral messages and are therefore independent of $\rho$.}",
     r"\label{tabres}", r"\centering\footnotesize", r"\begin{tabular}{llccccc}", r"\toprule",
     r"$\rho$ & Config. & Delivered ratio & Mean wait [min] & P95 wait [min] & Vuln. wait [min] & PC \\", r"\midrule"]
for rho in (0.0, 0.2, 0.4):
    first = True
    for cfg in ["B0", "B1", "B2", "B3", "B4", "B5", "O1"]:
        g = main[(main.p_rho == rho) & (main.config == cfg)]
        L.append(f"{rho:g} & " if first else " & ")
        L[-1] += f"{cfg} & {pm(g.delivered_ratio.mean(), g.delivered_ratio.std(), 3)} & {pm(g.wait_mean.mean(), g.wait_mean.std(), 1)} & {pm(g.wait_p95.mean(), g.wait_p95.std(), 1)} & {pm(g.wait_vuln.mean(), g.wait_vuln.std(), 1)} & {pm(g.pc.mean(), g.pc.std(), 3)} \\\\"
        first = False
    L.append(r"\midrule" if rho != 0.4 else r"\midrule")
first = True
for cfg in ["B0-NR", "B3-NR", "B4-scratch"]:
    g = main[(main.p_rho == 0.2) & (main.config == cfg)]
    L.append(("any & " if first else " & ") + f"{cfg} & {pm(g.delivered_ratio.mean(), g.delivered_ratio.std(), 3)} & {pm(g.wait_mean.mean(), g.wait_mean.std(), 1)} & {pm(g.wait_p95.mean(), g.wait_p95.std(), 1)} & {pm(g.wait_vuln.mean(), g.wait_vuln.std(), 1)} & {pm(g.pc.mean(), g.pc.std(), 3)} \\\\")
    first = False
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
write("tab_results", "\n".join(L) + "\n")

# ---------------------------------------------------------------- contrasts
NAMES = {"B3 - B1": r"B3$-$B1 (authenticated vs claims, det.)", "B4 - B2": r"B4$-$B2 (authenticated vs claims, learned)",
         "B4 - B3": r"B4$-$B3 (learned vs deterministic)", "B4 - B5": r"B4$-$B5 (decentralised vs centralised)",
         "B3 - B0": r"B3$-$B0 (priority scheduling vs FCFS)"}
MET = [("delivered_ratio", 3), ("wait_p95", 2), ("pc", 3), ("wait_vuln", 2)]
L = [r"\begin{table*}[!t]", r"\caption{Paired differences A$-$B over 30 test seeds: mean [95\% CI]. $\dagger$: significant after Holm correction over the 60 primary tests (adjusted $p<0.05$). Negative waiting-time differences favour A.}",
     r"\label{tabcon}", r"\centering\footnotesize", r"\begin{tabular}{llcccc}", r"\toprule",
     r"$\rho$ & Contrast (A$-$B) & $\Delta$ Delivered ratio & $\Delta$ P95 wait [min] & $\Delta$ PC & $\Delta$ Vuln. wait [min] \\", r"\midrule"]
for rho in (0.0, 0.2, 0.4):
    first = True
    for cn in ["B3 - B1", "B4 - B2", "B4 - B3", "B4 - B5", "B3 - B0"]:
        row = [f"{rho:g}" if first else "", NAMES[cn]]
        for m, nd in MET:
            r = con[(con.rho == rho) & (con.contrast == cn) & (con.metric == m)].iloc[0]
            dag = r"$^\dagger$" if r.p_holm < 0.05 else ""
            row.append(f"{n(r['diff'], nd, True)} [{n(r.ci_lo, nd, True)}, {n(r.ci_hi, nd, True)}]{dag}")
        L.append(" & ".join(row) + r" \\")
        first = False
    if rho != 0.4:
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
write("tab_contrasts", "\n".join(L) + "\n")

# ---------------------------------------------------------------- waiting time by group / route
cfgs = ["B0", "B1", "B3", "B4", "B5", "O1"]
rows = [("All requests", "all", "all"), ("General", "group", "general"), ("Elderly", "group", "elderly"), ("Medical", "group", "medical"),
        ("Token", "route", "token"), ("Face", "route", "face"), ("Face failure $\\rightarrow$ token", "route", "face_fb"), ("Staff assistance", "route", "assist")]
L = [r"\begin{table*}[!t]", r"\caption{Mean waiting time [min] by eligibility group and credential route at $\rho=0.2$ (mean over 30 test seeds). $n$: mean number of requests per run. Group rows and route rows are two separate breakdowns of the same requests; the four route rows cover vulnerable users only. Unpaired 95\% CI half-widths of the means are 1.6--1.8 min (3.0 min for face failure $\rightarrow$ token); differences between configurations are paired and much narrower.}",
     r"\label{tabgrp}", r"\centering\footnotesize", r"\begin{tabular}{lrcccccc}", r"\toprule",
     r"Subset & $n$ per run & B0 & B1 & B3 & B4 & B5 & O1 \\", r"\midrule"]
hw = []
for lab, kind, key in rows:
    vals, nn = [], None
    for cfg in cfgs:
        g = B[(B.config == cfg) & (B.kind == kind) & (B.key == key)]
        nn = g.n.mean()
        v = g.wait_mean.dropna()
        vals.append(f"{v.mean():.1f}")
        hw.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) * 1.0)
    L.append(f"{lab} & {nn:.0f} & " + " & ".join(vals) + r" \\")
    if lab in ("All requests", "Medical"):
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
write("tab_groups", "\n".join(L) + "\n")
print("max CI half-width (waiting-time table):", round(max(hw), 2))

# ---------------------------------------------------------------- fallback by reason
reasons = [("Face failure $\\rightarrow$ token", "face_fail_token"), ("Not enrolled $\\rightarrow$ staff", "not_enrolled"),
           ("Token lost $\\rightarrow$ staff", "token_lost"), ("Face failure, no token $\\rightarrow$ staff", "face_fail_no_token"),
           ("All staff-assisted", "all_assistance"), ("No fallback (vulnerable)", "no_fallback_vuln")]
L = [r"\begin{table}[!t]", r"\caption{Fallback pathways at $\rho=0.2$ (mean over 30 test seeds). Delay: time until the authorised priority becomes effective. Priority: share of requests granted any eligibility priority / priority at least equal to the true eligibility score. Completed and wait: B3 (authenticated) vs O1 (oracle identity).}",
     r"\label{tabfb}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r" & $n$ & Delay & \multicolumn{2}{c}{Priority [\%]} & \multicolumn{2}{c}{Completed [\%]} & Wait [min] \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"Reason & per run & [min] & any & full & B3 & O1 & B3 / O1 \\", r"\midrule"]
for lab, key in reasons:
    a = RS[(RS.config == "B3") & (RS.key == key)].iloc[0]; o = RS[(RS.config == "O1") & (RS.key == key)].iloc[0]
    L.append(f"{lab} & {a.n_per_seed:.1f} & {a.delay_min:.1f} & {100*a.prio_granted:.0f} & {100*a.prio_full:.0f} & {100*a.completed:.1f} & {100*o.completed:.1f} & {a.wait_mean:.1f} / {o.wait_mean:.1f} \\\\")
    if key == "face_fail_no_token":
        pass
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("tab_fallback", "\n".join(L) + "\n")

# ---------------------------------------------------------------- failure recovery
KN = {"outage": "Equipment outage (4 of 20)", "stockout": "Stock-out (4 of 20)", "partition": "Network partition",
      "pol_crash": "Policy crash $\\rightarrow$ local rule", "pol_random": "Random policy proposals"}
L = [r"\begin{table*}[!t]", r"\caption{Failure recovery at $\rho=0.2$. Each fault lasts 60 min and starts at a random time between 2 h and 4 h into the 8~h shift. Recovery: time after the fault ends until the backlog excess over the fault-free twin stays within 10\% of the mean backlog for 10 min (median [IQR]); a value of 0 means no measurable degradation. Deficit: loss of delivered volume relative to the fault-free twin (mean [95\% CI]). Not rec.: runs not recovered by the end of the 8~h horizon (of 30).}",
     r"\label{tabflt}", r"\centering\footnotesize", r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Fault & Config. & Recovery [min] & Peak backlog excess & Deficit [\%] & $\Delta$ wait [min] & $\Delta$ aband. [pp] & Not rec. \\", r"\midrule"]
for kind in ["outage", "stockout", "partition", "pol_crash", "pol_random"]:
    d = fsum[fsum.fault == kind]
    first = True
    for cfg in ["B0", "B3", "B4", "B5"]:
        r = d[d.config == cfg]
        if r.empty:
            continue
        r = r.iloc[0]
        L.append((KN[kind] if first else "") + f" & {cfg} & {r.rec_median:.0f} [{r.rec_q1:.0f}, {r.rec_q3:.0f}] & {r.peak_mean:.1f} & {n(r.deficit_mean,2)} [{n(r.deficit_lo,2)}, {n(r.deficit_hi,2)}] & {n(r.dwait_mean, 2, True)} & {n(r.dabandon_mean, 2, True)} & {int(r.censored)} \\\\")
        first = False
    L.append(r"\midrule" if kind != "pol_random" else "")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
write("tab_faults", "\n".join(l for l in L if l != "") + "\n")

# ---------------------------------------------------------------- enrolment sensitivity
s = raw[raw.tag == "sens_c_enr"]
m = raw[(raw.tag == "main") & (raw.p_rho == 0.2)]
def row(df):
    return df.pc.mean(), 100 * df.vuln_missed_priority.mean(), df.wait_vuln.mean(), 100 * df.unauth_rate.mean()
L = [r"\begin{table}[!t]", r"\caption{Sensitivity to enrolment coverage $c_{\mathrm{enr}}$ at $\rho=0.2$ (mean over 30 test seeds). Below true priority: vulnerable requests whose effective priority score is below their true eligibility score, including temporary staff-assigned priority. Unauth.: requests with priority above the true eligibility.}",
     r"\label{tabenr}", r"\centering\footnotesize", r"\begin{tabular}{llcccc}", r"\toprule",
     r"Config. & $c_{\mathrm{enr}}$ & PC & Below true priority [\%] & Vuln. wait [min] & Unauth. [\%] \\", r"\midrule"]
for c in (0.4, 0.6, 0.8, 1.0):
    df = m[m.config == "B3"] if c == 0.8 else s[(s.config == "B3") & (s.p_c_enr == c)]
    a = row(df)
    L.append(f"B3 & {c:.1f} & {a[0]:.3f} & {a[1]:.1f} & {a[2]:.1f} & {a[3]:.1f} \\\\")
L.append(r"\midrule")
a = row(m[m.config == "B1"])
L.append(f"B1 (claims) & -- & {a[0]:.3f} & {a[1]:.1f} & {a[2]:.1f} & {a[3]:.1f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("tab_enrol", "\n".join(L) + "\n")
print("done")
