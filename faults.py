"""Failure-recovery experiments (Table III row "Failure recovery") and the per-reason / per-route
breakdowns (Table III rows "Waiting time", "Fallback completion").

Everything runs on the TEST seeds 9000-9029 at rho = 0.2 (independent loss).  Faults are drawn from a
SEPARATE generator (seed + 7_000_003), so the request stream, equipment failures, replenishment and
packet-loss realisation of every episode are exactly those of the main experiment; a faulted episode
differs from its fault-free twin only through the injected fault (matched pairs).

Fault types (window = 60 min = 30 intervals, start drawn uniformly in [0.25 H, 0.50 H)):
  outage     4 of 20 dispensers (20 %) are down (equipment fault).
  stockout   the tanks of 4 dispensers are empty for the window (stock lost, deliveries blocked); one
             emergency resupply of 1000 L arrives at the end of the window.
  partition  the network splits into two halves (west / east of the median x-coordinate); the central
             node of B5 stays with the west half.
  pol_crash  (B4 only) the learned policy stops producing output; each dispenser continues with the local
             deterministic scheduler (priority + aging, no referral) as specified in Section VI-D.
  pol_random (B4 only) the policy emits uniformly random proposals; only the safety supervisor protects.

Recovery metric.  Q_f(k), Q_0(k): number of queued or in-transit requests in the faulted / fault-free twin.
dQ(k) = Q_f(k) - Q_0(k); theta = max(2, 0.10 * mean_k Q_0(k)).  Recovery time = time from the end of the
fault window until dQ stays <= theta for 5 consecutive intervals (10 min); runs that never do so by the end
of the horizon are right-censored (flag `censored`, value = remaining horizon).
"""
import sys, json, time, copy
import numpy as np, pandas as pd
from multiprocessing import Pool
from sim import *
import run_eval as R

TEST = R.TEST
WIN = 30
CONFIGS = ["B0", "B3", "B4", "B5"]
FAULTS = ["outage", "stockout", "partition"]
RHO = 0.2
ROUTES = ("token", "face", "face_fb", "assist", "none", "misuse")
REASONS = {1: "face_fail_token", 2: "not_enrolled", 3: "token_lost", 4: "face_fail_no_token"}


def plan(seed, n):
    rng = np.random.default_rng(seed + 7_000_003)
    k0 = int(rng.uniform(0.25, 0.50) * H)
    idx = np.sort(rng.choice(n, max(1, round(0.2 * n)), replace=False))
    return k0, k0 + WIN, idx


def faulted_scenario(sc, kind, k0, k1, idx):
    s2 = copy.copy(sc)
    if kind == "outage":
        s2.up = sc.up.copy()
        s2.up[k0:k1, idx] = False
    elif kind == "stockout":
        s2.stockout = (k0, k1, idx)
    elif kind == "partition":
        s2.part = (k0, k1)
    return s2


class PolicyFault:
    """Wraps a learned controller; inside [k0, k1) its output is replaced (see module docstring)."""
    def __init__(self, base, mode, k0, k1, seed):
        self.base, self.mode, self.k0, self.k1 = base, mode, k0, k1
        self.timed, self.uses_centre = base.timed, base.uses_centre
        self.fb = Heur(use_ref=False)
        self.rng = np.random.default_rng(seed + 9_000_011)

    def act(self, sim):
        if self.k0 <= sim.k < self.k1:
            if self.mode == "crash":
                return self.fb.act(sim)
            return self.rng.integers(0, ACT_DIM, sim.n).tolist()
        return self.base.act(sim)


def run_sim(sc, regime, ctl):
    sim = Sim(sc, regime)
    sim.run(ctl)
    return sim, sim.metrics()


def recovery(q0, qf, k0, k1):
    q0, qf = np.asarray(q0), np.asarray(qf)
    dq = qf - q0
    theta = max(2.0, 0.10 * q0.mean())
    peak = float(dq[k0:].max())
    krec, cens = None, False
    for k in range(k1, H):
        seg = dq[k:k + 5]
        if len(seg) == 5 and seg.max() <= theta:
            krec = k
            break
    if krec is None:
        cens = True
        T = (H - k1) * DT
    else:
        T = (krec - k1) * DT
    return dict(rec_min=T, censored=cens, peak_excess=peak, theta=theta,
                excess_area=float(np.clip(dq[k0:], 0, None).sum() * DT / 60.0))      # request-hours of excess backlog


FIELDS = ["delivered_ratio", "completed_rate", "abandon_rate", "wait_mean", "wait_p95", "wait_vuln",
          "pc", "protq_viol", "override", "n_comp", "n_viol", "offers", "offers_ok"]


def fault_job(a):
    cfg, seed = a
    regime, mk = R.make(cfg)
    P = Params(rho=RHO)
    sc = Scenario(seed, P)
    k0, k1, idx = plan(seed, P.n)
    sim0, m0 = run_sim(sc, regime, mk())
    rows = []
    d0 = sum(sim0.dhist)
    kinds = [(k, None) for k in FAULTS]
    if cfg == "B4":
        kinds += [("pol_crash", "crash"), ("pol_random", "random")]
    for kind, mode in kinds:
        if kind in FAULTS:
            sc2 = faulted_scenario(sc, kind, k0, k1, idx)
            ctl = mk()
        else:
            sc2 = sc
            ctl = PolicyFault(mk(), mode, k0, k1, seed)
        sim, m = run_sim(sc2, regime, ctl)
        r = dict(config=cfg, seed=seed, fault=kind, k0=k0, k1=k1)
        r.update(recovery(sim0.qhist, sim.qhist, k0, k1))
        r["deficit_delivered"] = 1.0 - sum(sim.dhist) / d0
        for f in FIELDS:
            r[f] = m[f]
            r["d_" + f] = m[f] - m0[f]
        dm = np.array(sim.dmat)
        r["deliv_from_faulty_L"] = float(dm[k0:k1, idx].sum()) if kind in ("outage", "stockout") else np.nan
        r["deliv_while_down"] = float(dm[~sc2.up[:H]].sum())      # safety invariant: must be exactly 0
        r["ovr_serve_infeasible"] = sim.override_reason.get("serve_infeasible", 0)
        rows.append(r)
    base = dict(config=cfg, seed=seed, fault="none", k0=k0, k1=k1)
    base.update({f: m0[f] for f in FIELDS})
    rows.append(base)
    return rows


# ------------------------------------------------------------------ breakdown tables
def breakdown(sim, cfg, seed):
    tend = H * DT
    reqs = sim.allr
    g = np.array([r.g for r in reqs]); route = np.array([r.au_route for r in reqs]); reason = np.array([r.reason for r in reqs])
    vol = np.array([r.vol for r in reqs]); srv = np.array([r.served for r in reqs])
    comp = np.array([r.done is not None for r in reqs])
    started = np.array([r.first is not None for r in reqs])
    exit_t = np.array([(r.first if r.first is not None else (r.gone if r.gone is not None else tend)) for r in reqs])
    wait = np.maximum(exit_t - np.array([r.t for r in reqs]), 0)
    delay = np.array([r.au_delay * DT for r in reqs])
    b_au = np.array([r.au_b for r in reqs]); bt = np.array([r.bt for r in reqs])
    vuln = g > 0
    rows = []

    def add(kind, key, m):
        if not m.any():
            rows.append(dict(config=cfg, seed=seed, kind=kind, key=key, n=0)); return
        rows.append(dict(config=cfg, seed=seed, kind=kind, key=key, n=int(m.sum()),
                         completed=float(comp[m].mean()), started=float(started[m].mean()),
                         wait_mean=float(wait[m].mean()), wait_p95=float(np.percentile(wait[m], 95)),
                         delivered_ratio=float(srv[m].sum() / vol[m].sum()), delay_min=float(delay[m].mean()),
                         prio_granted=float((b_au[m] > 0).mean()), prio_full=float((b_au[m] >= bt[m] - 1e-9).mean())))
    add("all", "all", np.ones(len(reqs), bool))
    for gi, gn in enumerate(GROUPS):
        add("group", gn, g == gi)
    add("group", "vulnerable", vuln)
    for rid, rn in enumerate(("token", "face", "face_fb", "assist", "none", "misuse")):
        add("route", rn, route == rid)
        add("route_vuln", rn, (route == rid) & vuln)
    for rc, rn in REASONS.items():
        add("reason", rn, reason == rc)
    add("reason", "no_fallback_vuln", (reason == 0) & vuln)
    add("reason", "all_assistance", route == 3)
    return rows


def breakdown_job(a):
    cfg, seed = a
    regime, mk = R.make(cfg)
    P = Params(rho=RHO)
    sim, m = run_sim(Scenario(seed, P), regime, mk())
    return breakdown(sim, cfg, seed)


def main():
    t0 = time.time()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = TEST if len(sys.argv) < 3 else TEST[:int(sys.argv[2])]
    if what in ("faults", "all"):
        jobs = [(c, s) for c in CONFIGS for s in seeds]
        with Pool(2) as p:
            res = p.map(fault_job, jobs, chunksize=1)
        df = pd.DataFrame([r for rr in res for r in rr])
        df.to_csv("results_faults.csv", index=False)
        print("faults", len(df), "rows", round(time.time() - t0), "s", flush=True)
    if what in ("breakdown", "all"):
        jobs = [(c, s) for c in ["B0", "B1", "B3", "B4", "B5", "O1"] for s in seeds]
        with Pool(2) as p:
            res = p.map(breakdown_job, jobs, chunksize=1)
        df = pd.DataFrame([r for rr in res for r in rr])
        df.to_csv("results_breakdown.csv", index=False)
        print("breakdown", len(df), "rows", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
