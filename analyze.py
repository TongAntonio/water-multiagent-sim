"""Statistics, tables and figures for Table I (configurations B0-B5, O1) from results_raw.csv.
Paired design: every seed is run under every configuration.  Reported: mean +- sd over 30 test seeds, paired
differences with 95% t-intervals, Cohen's d_z, paired-t p-values with Holm correction over the pre-declared
primary family (5 contrasts x 4 primary metrics x 3 packet-loss levels = 60 tests)."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("out", exist_ok=True)
df = pd.read_csv("results_raw.csv")
CORE = ["B0", "B1", "B2", "B3", "B4", "B5", "O1"]
SUPP = ["B3-NR", "B0-NR", "B4-scratch"]
COL = dict(zip(CORE, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})

PRIMARY = ["delivered_ratio", "wait_p95", "pc", "wait_vuln"]
CONTRASTS = [("B3", "B1", "RQ1: authenticated vs unverified eligibility (deterministic)"),
             ("B4", "B2", "RQ1: authenticated vs unverified eligibility (learned)"),
             ("B4", "B3", "RQ3: learned vs deterministic coordination"),
             ("B4", "B5", "learned decentralised vs centralised optimiser"),
             ("B3", "B0", "priority scheduling vs FCFS")]
LABEL = dict(delivered_ratio="Delivered-volume ratio", completed_rate="Completed requests", wait_mean="Mean wait [min]",
             wait_p95="P95 wait [min]", wait_vuln="Mean wait, vulnerable [min]", wait_general="Mean wait, general [min]", pc="Priority compliance PC",
             jain="Jain index J", unauth_rate="Unauthorised priority", frac_over_tprot="Wait >= T_protect")


def ci_t(x, a=0.05):
    x = np.asarray(x, float)
    m, s, n = x.mean(), x.std(ddof=1), len(x)
    h = stats.t.ppf(1 - a / 2, n - 1) * s / np.sqrt(n)
    return m, m - h, m + h


def msd(sub, col, fmt="{:.3f}"):
    x = sub[col].astype(float)
    return (fmt + " ± " + fmt).format(x.mean(), x.std(ddof=1))


# ---------------------------------------------------------------- main summary tables
main = df[df.tag == "main"]
rows = []
for rho in (0.0, 0.2, 0.4):
    for cfg in CORE + SUPP:
        s = main[(main.p_rho == rho) & (main.config == cfg)]
        r = dict(rho=rho, config=cfg, n_seeds=len(s))
        for c in ["delivered_ratio", "completed_rate", "abandon_rate", "wait_mean", "wait_p95", "wait_vuln", "wait_general", "wait_elderly", "wait_medical",
                  "pc", "jain", "x_general", "x_elderly", "x_medical", "unauth_rate", "vuln_missed_priority", "frac_over_tprot", "offers", "offers_ok",
                  "offer_accept_rate", "hs_fail", "override", "bytes_per_disp_hour", "mean_age", "ms_per_decision_step", "n_comp", "n_viol", "protq_viol",
                  "assist_delay_min", "assist_completion"]:
            v = s[c].astype(float)
            r[c + "_mean"] = v.mean(); r[c + "_sd"] = v.std(ddof=1)
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv("out/summary_main.csv", index=False)

# ---------------------------------------------------------------- paired contrasts + Holm
crow = []
for rho in (0.0, 0.2, 0.4):
    for a, b, why in CONTRASTS:
        A = main[(main.p_rho == rho) & (main.config == a)].set_index("seed")
        B = main[(main.p_rho == rho) & (main.config == b)].set_index("seed")
        for m in PRIMARY:
            d = (A[m] - B.loc[A.index, m]).astype(float)
            mean, lo, hi = ci_t(d)
            if d.std(ddof=1) < 1e-12:
                p, dz = 1.0, 0.0
            else:
                p = stats.ttest_rel(A[m], B.loc[A.index, m]).pvalue
                dz = d.mean() / d.std(ddof=1)
            crow.append(dict(rho=rho, contrast=f"{a} - {b}", rq=why, metric=m, diff=mean, ci_lo=lo, ci_hi=hi, dz=dz, p=p, n=len(d)))
con = pd.DataFrame(crow)
order = np.argsort(con.p.values)
m_ = len(con)
holm = np.empty(m_)
run = 0.0
for rank, idx in enumerate(order):
    run = max(run, min(1.0, (m_ - rank) * con.p.values[idx]))
    holm[idx] = run
con["p_holm"] = holm
con["significant_holm_0.05"] = con.p_holm < 0.05
con.to_csv("out/contrasts_primary.csv", index=False)

# ---------------------------------------------------------------- scenarios (burst / partition / scale)
sc = df[df.tag.isin(["main", "burst", "partition", "scale10", "scale40"])].copy()
sc = sc[~((sc.tag == "main") & (sc.p_rho != 0.2))]
sc["scenario"] = sc.tag.replace({"main": "iid loss 0.2"})
scen = sc[sc.config.isin(CORE)].groupby(["scenario", "config"])[["delivered_ratio", "wait_p95", "pc", "wait_vuln", "jain"]].agg(["mean", "std"])
scen.to_csv("out/summary_scenarios.csv")

# ---------------------------------------------------------------- sensitivity
sens = df[df.tag.str.startswith("sens_")].copy()
nominal = main[main.p_rho == 0.2]
sens_rows = []
for par in sorted({t[5:] for t in sens.tag.unique()}):
    sub = sens[sens.tag == f"sens_{par}"]
    vals = sorted(sub[f"p_{par}"].unique())
    for cfg in ["B1", "B2", "B3", "B4", "B5"]:
        for v in vals + [None]:
            s = sub[(sub.config == cfg) & (sub[f"p_{par}"] == v)] if v is not None else None
            if v is None:
                nomv = {"p_false": 0.08, "c_enr": 0.8, "fnmr": 0.05, "p_face": 0.5, "p_lost": 0.05, "p_acc": 0.7}[par]
                s = nominal[nominal.config == cfg]; v = nomv
            sens_rows.append(dict(param=par, value=v, config=cfg, wait_vuln=s.wait_vuln.mean(), wait_general=s.wait_general.mean(),
                                  pc=s.pc.mean(), unauth_rate=s.unauth_rate.mean(), delivered_ratio=s.delivered_ratio.mean(),
                                  vuln_missed=s.vuln_missed_priority.mean(), x_medical=s.x_medical.mean(), wait_p95=s.wait_p95.mean(), n=len(s)))
sens_df = pd.DataFrame(sens_rows).drop_duplicates(["param", "value", "config"]).sort_values(["param", "config", "value"])
sens_df.to_csv("out/sensitivity.csv", index=False)

# ---------------------------------------------------------------- figures
def bars(ax, metric, ylabel, rhos=(0.0, 0.2, 0.4), ylim=None):
    w = 0.115
    for ci, cfg in enumerate(CORE):
        xs, ys, es = [], [], []
        for ri, rho in enumerate(rhos):
            v = main[(main.p_rho == rho) & (main.config == cfg)][metric].astype(float)
            m, lo, hi = ci_t(v)
            xs.append(ri + (ci - 3) * (w + 0.012)); ys.append(m); es.append((hi - lo) / 2)
        ax.bar(xs, ys, w, color=COL[cfg], edgecolor="#fcfcfb", linewidth=1.2, label=cfg, yerr=es, error_kw=dict(ecolor=INK, lw=0.7, capsize=0))
    ax.set_xticks(range(len(rhos))); ax.set_xticklabels([f"ρ = {r:g}" for r in rhos])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
bars(axs[0], "delivered_ratio", "Delivered-volume ratio", ylim=(0.6, 1.0))
bars(axs[1], "wait_p95", "P95 waiting time [min]")
bars(axs[2], "wait_vuln", "Mean wait, vulnerable users [min]")
bars(axs[3], "pc", "Priority compliance PC", ylim=(0.8, 1.01))
for a, t in zip(axs, ["Delivered volume", "Tail waiting time", "Vulnerable users' wait", "Compliance with intended order"]):
    a.set_title(t, loc="left", fontsize=10, color=INK)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Table I configurations on 30 paired test seeds (mean, 95% CI)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig("out/fig1_main.png", dpi=170); plt.close(fig)

# group-specific outcomes at rho = 0.2 (RQ2)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (met, yl, ttl) in zip(axs, [("wait", "Mean wait [min]", "Waiting time by group"), ("x", "Demand satisfaction x_g", "Delivered / requested by group"),
                                    ("fm", "", "")]):
    pass
gl = ["general", "elderly", "medical"]
w = 0.115
for ax, kind, yl in [(axs[0], "wait", "Mean waiting time [min]"), (axs[1], "x", "Demand satisfaction x_g")]:
    for ci, cfg in enumerate(CORE):
        ys, es = [], []
        for g in gl:
            v = main[(main.p_rho == 0.2) & (main.config == cfg)][f"{kind}_{g}"].astype(float)
            m, lo, hi = ci_t(v); ys.append(m); es.append((hi - lo) / 2)
        ax.bar(np.arange(3) + (ci - 3) * (w + 0.012), ys, w, color=COL[cfg], edgecolor="#fcfcfb", linewidth=1.2, label=cfg, yerr=es,
               error_kw=dict(ecolor=INK, lw=0.7, capsize=0))
    ax.set_xticks(range(3)); ax.set_xticklabels(["general", "elderly / mobility", "medical needs"]); ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
axs[1].set_ylim(0.6, 1.0)
axs[0].set_title("Waiting time by group (ρ = 0.2)", loc="left", fontsize=10, color=INK)
axs[1].set_title("Demand satisfaction by group (ρ = 0.2)", loc="left", fontsize=10, color=INK)
# Jain index + unauthorised priority
ax = axs[2]
for ci, cfg in enumerate(CORE):
    v = main[(main.p_rho == 0.2) & (main.config == cfg)]["unauth_rate"].astype(float) * 100
    m, lo, hi = ci_t(v)
    ax.bar(ci, m, 0.7, color=COL[cfg], edgecolor="#fcfcfb", linewidth=1.2, yerr=(hi - lo) / 2, error_kw=dict(ecolor=INK, lw=0.7, capsize=0))
    ax.text(ci, m + 0.15, f"{m:.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(len(CORE))); ax.set_xticklabels(CORE); ax.set_ylabel("Requests with unauthorised priority [%]"); ax.grid(axis="x", visible=False)
ax.set_title("Unauthorised priority assignments", loc="left", fontsize=10, color=INK)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.4, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("out/fig2_groups.png", dpi=170); plt.close(fig)

# sensitivity of vulnerable users' wait
pars = [("p_false", "False-claim rate of general users"), ("c_enr", "Enrolment coverage"), ("fnmr", "Face false-non-match rate"),
        ("p_face", "Share of enrolled using the face route"), ("p_lost", "Credential lost / unusable"), ("p_acc", "Referral acceptance")]
fig, axs = plt.subplots(2, 3, figsize=(13, 6.2), sharey=True)
for ax, (par, ttl) in zip(axs.ravel(), pars):
    for cfg in ["B1", "B2", "B3", "B4", "B5"]:
        s = sens_df[(sens_df.param == par) & (sens_df.config == cfg)].sort_values("value")
        ax.plot(s.value, s.wait_vuln, "-o", color=COL[cfg], lw=2, ms=5, mfc=COL[cfg], mec="#fcfcfb", mew=1.2, label=cfg)
    ax.set_xlabel(ttl); ax.grid(axis="x", visible=False)
for ax in axs[:, 0]:
    ax.set_ylabel("Mean wait, vulnerable users [min]")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Sensitivity of vulnerable users' waiting time to identity assumptions (ρ = 0.2, 30 seeds per point)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig("out/fig3_sensitivity.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- markdown / latex tables
def fmt_row(rho, cfg, cols):
    s = summ[(summ.rho == rho) & (summ.config == cfg)].iloc[0]
    return [f"{s[c + '_mean']:.{d}f} ± {s[c + '_sd']:.{d}f}" for c, d in cols]

cols = [("delivered_ratio", 3), ("completed_rate", 3), ("wait_mean", 1), ("wait_p95", 1), ("wait_vuln", 1), ("pc", 3), ("jain", 4)]
hdr = ["Config", "Delivered ratio", "Completed", "Mean wait", "P95 wait", "Wait (vuln.)", "PC", "Jain J"]
with open("out/table_main.md", "w") as f:
    for rho in (0.0, 0.2, 0.4):
        f.write(f"\n**Packet loss ρ = {rho:g}** (mean ± sd over 30 paired test seeds)\n\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for cfg in CORE + SUPP:
            f.write("| " + cfg + " | " + " | ".join(fmt_row(rho, cfg, cols)) + " |\n")
for rho in (0.0, 0.2, 0.4):
    with open(f"out/table_main_rho{rho:g}.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\hline\n" + " & ".join(hdr) + " \\\\\\hline\n")
        for cfg in CORE:
            f.write(cfg + " & " + " & ".join(x.replace("±", "$\\pm$") for x in fmt_row(rho, cfg, cols)) + " \\\\\n")
        f.write("\\hline\\end{tabular}\n")

ct = con.copy()
ct["diff [95% CI]"] = ct.apply(lambda r: f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", axis=1)
with open("out/table_contrasts.md", "w") as f:
    for rho in (0.0, 0.2, 0.4):
        f.write(f"\n**ρ = {rho:g}** (paired differences A − B, n = 30; Holm-adjusted p over 60 tests)\n\n| Contrast | Metric | A − B [95% CI] | d_z | p_Holm |\n|---|---|---|---|---|\n")
        for _, r in ct[ct.rho == rho].iterrows():
            f.write(f"| {r.contrast} | {LABEL[r.metric]} | {r['diff [95% CI]']} | {r.dz:+.2f} | {r.p_holm:.3g} |\n")
print("ok")
print(summ[summ.rho == 0.2][["config", "delivered_ratio_mean", "wait_p95_mean", "wait_vuln_mean", "pc_mean", "jain_mean"]].round(3).to_string())
