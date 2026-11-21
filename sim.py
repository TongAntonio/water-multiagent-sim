"""
Discrete-event multi-agent simulator for the water-distribution framework
(Section VIII-C of the paper).  All constants below are fixed a priori and are
NOT tuned on test seeds.  Only the arrival intensity and the heuristic-referral
thresholds were calibrated on validation seeds (see calibrate.py).

Units: minutes, litres.  One decision interval = DT minutes; one dispenser can
perform at most ONE action per interval (Eq. 4): serve / wait / offer referral.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------- constants
DT = 2.0            # decision interval [min]
H = 240             # decisions per episode (8 h)
VMAX = 20.0         # per-actuation volume limit [L]
LMAX = 4000.0       # storage capacity [L]
VOLS = np.array([20.0, 40.0, 60.0])
VOL_P = np.array([0.50, 0.35, 0.15])
GROUPS = ("general", "elderly", "medical")
GROUP_P = np.array([0.72, 0.18, 0.10])
B_TRUE = np.array([0.0, 0.6, 1.0])        # eligibility score phi(h_u)
B_ASSIST = 0.5                            # temporary staff-assigned eligibility
W_B, W_W = 0.6, 0.4                       # Eq. 13 weights (geo + recent-service disabled -> renormalised)
T_AGE = 60.0                              # Eq. 14 [min]
T_PROT = 40.0                             # protected-queue threshold [min]
STALE_MAX = 5                             # staleness threshold [intervals]
K_NB = 3                                  # neighbours per dispenser
MAX_REFS = 2                              # max referrals per request
COOL_STEPS = 5                            # no re-offer for 10 min after a declined referral
SUM_BYTES, OFFER_BYTES, ACK_BYTES, CMD_BYTES, REPORT_BYTES = 40, 64, 32, 48, 48
OBS_DIM = 35
ACT_DIM = 2 + K_NB                        # 0 serve, 1 wait, 2.. offer(neighbour a)
A_SERVE, A_WAIT = 0, 1

REGIMES = ("fcfs", "claim", "auth", "oracle")


@dataclass
class Params:
    n: int = 20
    rho: float = 0.2
    loss_mode: str = "iid"        # iid | burst | partition
    base_rate: float = 0.18       # mean arrivals / min / dispenser (stress calibrated, see calibrate.py)
    rate_cv: float = 0.45
    n_surge: int = 3
    surge_mult: float = 2.5
    surge_len: float = 90.0
    p_acc: float = 0.7            # referral acceptance
    c_enr: float = 0.8            # enrolment coverage of vulnerable users
    p_face: float = 0.5           # share of enrolled who use the (optional) face route
    p_token_also: float = 0.7     # face users who also hold a token
    p_lost: float = 0.05          # credential lost / unusable
    fnmr: float = 0.05            # false non-match (sensitivity parameter)
    fta: float = 0.03             # failure to acquire
    fmr: float = 1e-3             # false match (sensitivity parameter)
    p_claim: float = 0.90         # vulnerable users who claim priority (unverified regime)
    p_false: float = 0.08         # general users who falsely claim priority
    p_misuse: float = 0.005       # general users presenting a copied credential
    assist_mean: float = 10.0     # staff assistance delay [min]
    p_staff_ok: float = 0.90
    patience_mean: float = 90.0
    km_side: float = 5.0
    walk_min_km: float = 12.0
    mean_up: float = 240.0
    down_lo: float = 10.0
    down_hi: float = 40.0
    repl_every: float = 150.0
    repl_lo: float = 800.0
    repl_hi: float = 1500.0
    init_lo: float = 600.0
    init_hi: float = 2500.0


# ------------------------------------------------------------------ scenario
class Req:
    __slots__ = ("id", "disp", "g", "bt", "t", "vol", "rem", "pat", "au_delay", "au_b", "au_route",
                 "claim_b", "u_acc", "arr_k", "first", "done", "gone", "refs", "prev", "served",
                 "nacc", "assisted", "cool", "reason")


class Scenario:
    """All random draws are made once from the seed so that every configuration
    sees the *same* request stream, failures, replenishment and packet-loss
    realisation (matched / paired design)."""

    def __init__(self, seed: int, P: Params):
        self.seed, self.P = seed, P
        rng = np.random.default_rng(seed)
        n = P.n
        self.pos = rng.uniform(0, P.km_side, (n, 2))
        d = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=2)
        np.fill_diagonal(d, np.inf)
        self.nb = np.argsort(d, axis=1)[:, :K_NB]
        dist = np.take_along_axis(d, self.nb, 1)
        self.travel = np.maximum(1, np.ceil(dist * P.walk_min_km / DT)).astype(int)
        s2 = math.log(1 + P.rate_cv ** 2)
        mult = rng.lognormal(-0.5 * s2, math.sqrt(s2), n)
        lam = np.tile(P.base_rate * mult, (H, 1))
        for i in rng.choice(n, min(P.n_surge, n), replace=False):
            s = rng.uniform(0, H * DT - P.surge_len)
            lam[int(s / DT):int((s + P.surge_len) / DT), i] *= P.surge_mult
        counts = rng.poisson(lam * DT)
        counts[H - 1] = 0
        M = int(counts.sum())
        disp = np.repeat(np.tile(np.arange(n), H), counts.reshape(-1))
        kk = np.repeat(np.repeat(np.arange(H), n), counts.reshape(-1))
        t = (kk + rng.random(M)) * DT
        order = np.argsort(t, kind="stable")
        disp, kk, t = disp[order], kk[order], t[order]
        vol = rng.choice(VOLS, M, p=VOL_P)
        g = rng.choice(3, M, p=GROUP_P)
        pat = t + rng.gamma(2.5, P.patience_mean / 2.5, M)
        vuln = g > 0
        U = lambda: rng.random(M)
        enrolled = vuln & (U() < P.c_enr)
        face = enrolled & (U() < P.p_face)
        token_also = U() < P.p_token_also
        lost = U() < P.p_lost
        face_fail = U() < (P.fnmr + P.fta)
        assist_steps = np.maximum(1, np.ceil(rng.exponential(P.assist_mean, M) / DT)).astype(int)
        staff_ok = U() < P.p_staff_ok
        au_b = np.zeros(M)
        au_delay = np.zeros(M, int)
        route = np.full(M, 4)                       # 0 token,1 face,2 face->token,3 assistance,4 none,5 misuse
        need_assist = vuln & ~enrolled
        tok = enrolled & ~face
        need_assist |= tok & lost
        route[tok & ~lost] = 0
        au_b[tok & ~lost] = B_TRUE[g][tok & ~lost]
        f_ok = face & ~face_fail
        route[f_ok] = 1
        au_b[f_ok] = B_TRUE[g][f_ok]
        f_fb = face & face_fail & token_also & ~lost
        route[f_fb] = 2
        au_b[f_fb] = B_TRUE[g][f_fb]
        au_delay[f_fb] = 1
        need_assist |= face & face_fail & ~(token_also & ~lost)
        route[need_assist] = 3
        au_delay[need_assist] = assist_steps[need_assist]
        au_b[need_assist] = np.where(staff_ok[need_assist], B_ASSIST, 0.0)
        # fallback reason (bookkeeping only; consumes no random numbers)
        reason = np.zeros(M, int)
        reason[f_fb] = 1                                        # face failure recovered through the token
        reason[vuln & ~enrolled] = 2                            # not enrolled -> staff assistance
        reason[tok & lost] = 3                                  # token lost -> staff assistance
        reason[face & face_fail & ~(token_also & ~lost)] = 4    # face failure, no usable token -> staff assistance
        # copied-credential misuse by general users (token-only records are not protected)
        mis = (g == 0) & (U() < P.p_misuse)
        prot = U() < P.p_face
        fm = U() < P.fmr
        accepted = mis & (~prot | fm)
        route[accepted] = 5
        pick_mis = np.where(rng.random(M) < 0.5, 0.6, 1.0)      # fixed number of draws: keeps streams matched across parameter sweeps
        au_b[accepted] = pick_mis[accepted]
        # unverified claims
        claim_b = np.zeros(M)
        cl = vuln & (U() < P.p_claim)
        claim_b[cl] = B_TRUE[g][cl]
        fcl = (g == 0) & (U() < P.p_false)
        pick_cl = np.where(rng.random(M) < 0.5, 0.6, 1.0)
        claim_b[fcl] = pick_cl[fcl]
        u_acc = rng.random((M, 3))
        self.arr = dict(disp=disp, kk=kk, t=t, vol=vol, g=g, pat=pat, au_b=au_b, au_delay=au_delay,
                        route=route, claim_b=claim_b, u_acc=u_acc, reason=reason)
        self.M = M
        # equipment & inventory
        up = np.ones((H, n), bool)
        repl = np.zeros((H, n))
        for i in range(n):
            tt = rng.exponential(P.mean_up) * rng.random()
            while tt < H * DT:
                dl = rng.uniform(P.down_lo, P.down_hi)
                up[int(tt / DT):int(min(H * DT, tt + dl) / DT) + 1, i] = False
                tt += dl + rng.exponential(P.mean_up)
            tt = rng.exponential(P.repl_every)
            while tt < H * DT:
                repl[int(tt / DT), i] += rng.uniform(P.repl_lo, P.repl_hi)
                tt += rng.exponential(P.repl_every)
        self.up, self.repl = up, repl
        self.L0 = rng.uniform(P.init_lo, P.init_hi, n)
        # packet loss realisation (before any retransmission; no retries configured)
        self.loss_nb = self._loss(rng, (H, n, K_NB))
        self.loss_hs = self._loss(rng, (H, n, 2))
        self.loss_c = self._loss(rng, (H, n, 2))
        xmed = np.median(self.pos[:, 0])
        self.side = (self.pos[:, 0] > xmed).astype(int)
        if P.loss_mode == "partition":
            k0 = int(rng.uniform(0.2, 0.6) * H)
            self.part = (k0, k0 + int(120 / DT))
        else:
            self.part = None

    def _loss(self, rng, shape):
        P = self.P
        if P.rho <= 0:
            return np.zeros(shape, bool)
        if P.loss_mode != "burst":
            return rng.random(shape) < P.rho
        # Gilbert-Elliott: bad state loses 90 %, mean burst 10 intervals, long-run loss rate rho
        pi_b = min(0.99, P.rho / 0.9)
        p_bg = 0.1
        p_gb = pi_b * p_bg / (1 - pi_b)
        links = int(np.prod(shape[1:]))
        state = rng.random(links) < pi_b
        out = np.zeros((shape[0], links), bool)
        for k in range(shape[0]):
            out[k] = state & (rng.random(links) < 0.9)
            flip = rng.random(links)
            state = np.where(state, flip >= p_bg, flip < p_gb)
        return out.reshape(shape)

    def blocked(self, k, i, j):
        """link i<->j cut by the partition window (j=-1: central node on side 0)."""
        if self.part is None or not (self.part[0] <= k < self.part[1]):
            return False
        sj = 0 if j < 0 else self.side[j]
        return self.side[i] != sj

    def build_requests(self):
        a = self.arr
        buckets = [[] for _ in range(H)]
        allr = []
        for idx in range(self.M):
            r = Req()
            r.id = idx
            r.disp = int(a["disp"][idx])
            r.g = int(a["g"][idx])
            r.bt = float(B_TRUE[r.g])
            r.t = float(a["t"][idx])
            r.vol = r.rem = float(a["vol"][idx])
            r.pat = float(a["pat"][idx])
            r.au_delay = int(a["au_delay"][idx])
            r.au_b = float(a["au_b"][idx])
            r.au_route = int(a["route"][idx])
            r.claim_b = float(a["claim_b"][idx])
            r.u_acc = a["u_acc"][idx]
            r.arr_k = int(a["kk"][idx]) + 1
            r.first = None
            r.done = None
            r.gone = None
            r.refs = 0
            r.prev = -1
            r.served = 0.0
            r.nacc = 0
            r.assisted = r.au_route == 3
            r.cool = -1
            r.reason = int(a["reason"][idx])
            buckets[int(a["kk"][idx])].append(r)
            allr.append(r)
        return buckets, allr


# --------------------------------------------------------------------- sim
class Sim:
    def __init__(self, sc: Scenario, regime: str, rew_w=(1.0, 0.6, 0.3, 0.3, 0.2)):
        assert regime in REGIMES
        self.sc, self.P, self.regime = sc, sc.P, regime
        n = self.n = sc.P.n
        self.nb, self.travel = sc.nb, sc.travel
        self.buckets, self.allr = sc.build_requests()
        self.queues = [[] for _ in range(n)]
        self.L = sc.L0.copy()
        self.k = 0
        self.transit = {}
        self.fc = np.zeros(n)                       # EWMA arrival volume per interval
        self.arrived_vol = np.zeros(n)
        # neighbour views (last received summary)
        self.v_L = np.tile(self.L[self.nb] / 1.0, 1)
        self.v_up = np.ones((n, K_NB))
        self.v_Q = np.zeros((n, K_NB))
        self.v_step = np.zeros((n, K_NB), int)
        # centre view (for the centralised baseline)
        self.c_L = self.L.copy(); self.c_up = np.ones(n); self.c_Q = np.zeros(n)
        self.c_rem = np.zeros(n); self.c_has = np.zeros(n); self.c_step = np.zeros(n, int)
        self.cand = [None] * n
        self.can_serve = np.zeros(n, bool)
        # accumulators
        self.bytes = 0.0
        self.n_comp = 0; self.n_viol = 0; self.n_protviol = 0
        self.offers = 0; self.offers_ok = 0; self.hs_fail = 0; self.nack = 0; self.declined = 0
        self.override = 0; self.override_reason = {}
        self.age_sum = 0.0; self.age_n = 0
        self.compute_s = 0.0; self.decisions = 0
        self.req_g = np.zeros(3); self.del_g = np.zeros(3)
        self.rw = rew_w
        self.delivered_total = 0.0
        self.travel_started = 0.0
        # per-interval histories and fault bookkeeping (no random numbers are drawn)
        self.qhist = []; self.dhist = []; self.dmat = []

    # ---- priorities
    def beff(self, r, k):
        reg = self.regime
        if reg == "fcfs":
            return 0.0
        if reg == "claim":
            return r.claim_b
        if reg == "oracle":
            return r.bt
        return r.au_b if (k - r.arr_k) >= r.au_delay else 0.0

    def unresolved(self, r, k):
        return self.regime == "auth" and (k - r.arr_k) < r.au_delay

    def select(self, q, k, t):
        if not q:
            return None
        if self.regime == "fcfs":
            return min(q, key=lambda r: (r.t, r.id))
        best_p = None
        best = None
        bp = -1.0
        for r in q:
            w = t - r.t
            if w >= T_PROT:
                if best_p is None or (r.t, r.id) < (best_p.t, best_p.id):
                    best_p = r
            else:
                P = W_B * self.beff(r, k) + W_W * (w / T_AGE)
                if best is None or P > bp + 1e-12 or (abs(P - bp) <= 1e-12 and (r.t, r.id) < (best.t, best.id)):
                    best, bp = r, P
        return best_p if best_p is not None else best

    def prio(self, r, k, t):
        return W_B * self.beff(r, k) + W_W * min(1.0, max(0.0, (t - r.t) / T_AGE))

    # ---- observations for learned policy
    def obs(self):
        n, k, t = self.n, self.k, self.k * DT
        X = np.zeros((n, OBS_DIM), np.float32)
        for i in range(n):
            q = self.queues[i]
            nq = len(q)
            x = X[i]
            x[0] = self.L[i] / LMAX
            x[1] = float(self.sc.up[k, i])
            x[2] = min(nq, 30) / 30.0
            x[3] = min(self.fc[i] * 5.0 / (5 * VMAX), 3.0)
            if nq:
                pm = 0.0; wm = 0.0; nun = 0; npr = 0; nas = 0
                for r in q:
                    w = t - r.t
                    p = self.prio(r, k, t)
                    pm = max(pm, p); wm = max(wm, w)
                    nun += self.unresolved(r, k)
                    npr += w >= T_PROT
                    nas += (self.regime == "auth" and r.assisted and (k - r.arr_k) < r.au_delay)
                x[4] = pm
                x[5] = min(wm / T_AGE, 2.0)
                x[6] = nun / nq
                x[7] = npr / nq
                x[8] = nas / nq
            c = self.cand[i]
            if c is not None:
                x[9] = 1.0
                x[10] = c.rem / 60.0
                x[11] = self.prio(c, k, t)
                x[12] = min((t - c.t) / T_AGE, 2.0)
                x[13] = float((t - c.t) >= T_PROT)
                x[14] = c.refs / MAX_REFS
                x[34] = float(k < c.cool)
            x[15] = float(self.can_serve[i])
            for a in range(K_NB):
                age = k - self.v_step[i, a]
                if age <= STALE_MAX:
                    o = 16 + 6 * a
                    x[o] = 1.0
                    x[o + 1] = age / STALE_MAX
                    x[o + 2] = self.v_L[i, a] / LMAX
                    x[o + 3] = min(self.v_Q[i, a], 30) / 30.0
                    x[o + 4] = self.v_up[i, a]
                    x[o + 5] = self.travel[i, a] / 10.0
        return X

    # ---- reference deterministic referral rule (used by B0/B1/B3 and as supervisor safe action)
    def best_neighbour(self, i, c, overload, thqj):
        best, bs = -1, 1e9
        need = min(c.rem, VMAX)
        for a in range(K_NB):
            age = self.k - self.v_step[i, a]
            if age > STALE_MAX or not self.v_up[i, a] or self.v_L[i, a] < need:
                continue
            if self.nb[i, a] == c.prev:
                continue
            if overload and self.v_Q[i, a] > thqj:
                continue
            s = self.v_Q[i, a] + 0.25 * self.travel[i, a]
            if s < bs:
                best, bs = a, s
        return best

    # ---- one decision step (phase 2: controller + supervisor + execution)
    def decide_and_execute(self, controller):
        """Run controller, apply supervisor, execute, advance clock."""
        sc, P, n = self.sc, self.P, self.n
        k = self.k
        t = k * DT
        q = self.queues
        t0 = time.perf_counter()
        acts = controller.act(self)
        dt_c = time.perf_counter() - t0
        if getattr(controller, "timed", False):
            self.compute_s += dt_c
            self.decisions += 1
        cmd_bytes = 0.0
        if getattr(controller, "uses_centre", False):
            cmd_bytes = controller.cmd_bytes
        D = np.zeros(n)
        hs_i = np.zeros(n); trv_i = np.zeros(n); ovr_i = np.zeros(n)
        comp_k = 0; viol_k = 0
        bytes_k = float(cmd_bytes)
        hs_bytes_k = 0.0
        travel_k = 0.0
        stat_n = n * K_NB if not getattr(controller, "uses_centre", False) else 0
        bytes_k += stat_n * SUM_BYTES
        if getattr(controller, "uses_centre", False):
            bytes_k += n * REPORT_BYTES
        for i in range(n):
            c = self.cand[i]
            a = int(acts[i])
            # ---- safety supervisor (deterministic constraints) ----
            reason = None
            if a == A_SERVE and not self.can_serve[i]:
                reason = "serve_infeasible"
            elif a >= 2:
                aa = a - 2
                if c is None:
                    reason = "offer_no_candidate"
                elif c.refs >= MAX_REFS:
                    reason = "offer_max_refs"
                elif k - self.v_step[i, aa] > STALE_MAX:
                    reason = "offer_stale_destination"
                elif self.nb[i, aa] == c.prev:
                    reason = "offer_back_to_origin"
                elif k < c.cool:
                    reason = "offer_cooldown"
            elif a == A_SERVE and c is None:
                reason = "serve_no_candidate"
            if reason is not None:
                self.override += 1
                ovr_i[i] = 1.0
                self.override_reason[reason] = self.override_reason.get(reason, 0) + 1
                a = A_SERVE if (self.can_serve[i] and c is not None) else A_WAIT
            if c is None:
                continue
            if a == A_SERVE:
                d = min(c.rem, VMAX, self.L[i])
                # priority-compliance bookkeeping (declared ordering uses ground-truth eligibility)
                qi = q[i]
                if len(qi) > 1:
                    ks = self._dkey(c, t)
                    nv = 0
                    for r in qi:
                        if r is not c and self._dkey(r, t) < ks:
                            nv += 1
                    comp_k += len(qi) - 1
                    viol_k += nv
                    # protected-queue ordering sanity check
                    if (t - c.t) < T_PROT and any((t - r.t) >= T_PROT for r in qi):
                        self.n_protviol += 1
                D[i] = d
                c.rem -= d
                c.served += d
                self.del_g[c.g] += d
                if c.first is None:
                    c.first = t
                if c.rem <= 1e-9:
                    c.done = t + DT
                    qi.remove(c)
            elif a >= 2:
                aa = a - 2
                j = int(self.nb[i, aa])
                self.offers += 1
                hs_bytes_k += OFFER_BYTES
                hs_i[i] += OFFER_BYTES
                lost1 = sc.loss_hs[k, i, 0] or sc.blocked(k, i, j)
                if lost1:
                    self.hs_fail += 1
                else:
                    hs_bytes_k += ACK_BYTES
                    hs_i[i] += ACK_BYTES
                    lost2 = sc.loss_hs[k, i, 1] or sc.blocked(k, i, j)
                    if lost2:
                        self.hs_fail += 1
                    elif not (sc.up[k, j] and self.L[j] >= min(c.rem, VMAX)):
                        self.nack += 1
                    elif c.u_acc[min(c.nacc, 2)] >= P.p_acc:
                        c.nacc += 1
                        c.cool = k + COOL_STEPS
                        self.declined += 1
                    else:
                        self.offers_ok += 1
                        q[i].remove(c)
                        c.refs += 1
                        c.prev = i
                        self.transit.setdefault(k + 1 + int(self.travel[i, aa]), []).append((j, c))
                        c.disp = j
                        travel_k += self.travel[i, aa]
                        trv_i[i] += self.travel[i, aa]
                        self.travel_started += self.travel[i, aa]
        self.bytes += bytes_k + hs_bytes_k
        # 5 inventory update and neighbour messages (sent at end of interval, describe state at k+1)
        self.L = np.maximum(0.0, self.L - D)
        self.delivered_total += D.sum()
        self.n_comp += comp_k
        self.n_viol += viol_k
        up_next = sc.up[min(k + 1, H - 1)]
        for i in range(n):
            for a in range(K_NB):
                j = self.nb[i, a]
                if not (sc.loss_nb[k, i, a] or sc.blocked(k, i, j)):
                    self.v_L[i, a] = self.L[j]
                    self.v_up[i, a] = float(up_next[j])
                    self.v_Q[i, a] = len(q[j])
                    self.v_step[i, a] = k + 1
        if getattr(controller, "uses_centre", False):
            for i in range(n):
                if not (sc.loss_c[k, i, 0] or sc.blocked(k, i, -1)):
                    self.c_L[i] = self.L[i]; self.c_up[i] = float(up_next[i]); self.c_Q[i] = len(q[i])
                    self.c_step[i] = k + 1
                    c2 = self.select(q[i], k + 1, (k + 1) * DT)
                    self.c_has[i] = float(c2 is not None)
                    self.c_rem[i] = c2.rem if c2 is not None else 0.0
        # staleness statistics
        for i in range(n):
            for a in range(K_NB):
                self.age_sum += (k + 1) - self.v_step[i, a]
                self.age_n += 1
        # reward components (training only; use ground truth).  In-transit requests still wait.
        trans_cnt = np.zeros(n)
        for lst in self.transit.values():
            for (_j, r_) in lst:
                trans_cnt[r_.prev] += 1
        qlen = np.array([len(x) for x in q], float)
        V_i = D / (n * VMAX)
        W_i = (qlen + trans_cnt) / (n * 10.0)
        C_i = hs_i / (n * 100.0 * n) * n + trv_i * DT / (n * 30.0)   # hs_i/(n*100)
        C_i = hs_i / (n * 100.0) + trv_i * DT / (n * 30.0)
        B = (viol_k / comp_k) if comp_k > 0 else 0.0
        m = self.req_g > 0
        if m.any():
            x = self.del_g[m] / self.req_g[m]
            Jn = (x.sum() ** 2) / (m.sum() * (x ** 2).sum()) if (x ** 2).sum() > 0 else 1.0
        else:
            Jn = 1.0
        F = 1.0 - Jn
        lv, lw, lb, lf, lc = self.rw
        r_i = lv * V_i - lw * W_i - lc * C_i - (lb * B + lf * F) / n      # sums to the team reward of Eq. (16)
        rew = float(r_i.sum())
        self.agent_rew = r_i
        self.agent_ovr = ovr_i
        self.qhist.append(float(qlen.sum() + trans_cnt.sum()))
        self.dhist.append(float(D.sum()))
        self.dmat.append(D.copy())
        self.k += 1
        return rew, self.k >= H

    def _dkey(self, r, t):
        w = t - r.t
        if w >= T_PROT:
            return (0, r.t, r.id)
        return (1, -(W_B * r.bt + W_W * w / T_AGE), r.t, r.id)

    # ---- episode driver
    def begin_step(self):
        """Phase 1 of a step: arrivals, abandonment, candidates (called before observation)."""
        k = self.k
        t = k * DT
        n = self.n
        q = self.queues
        self.L = np.minimum(LMAX, self.L + self.sc.repl[k])
        so = getattr(self.sc, "stockout", None)      # (k0, k1, idx): stock-out fault
        if so is not None:
            k0_, k1_, idx_ = so
            if k0_ <= k < k1_:                       # tank empty and deliveries blocked during the window
                self.L[idx_] = 0.0
            elif k == k1_:                           # one emergency resupply of 1000 L (mid-range of repl_lo..repl_hi)
                self.L[idx_] = np.minimum(LMAX, self.L[idx_] + 1000.0)
        add = np.zeros(n)
        if k > 0:
            for r in self.buckets[k - 1]:
                q[r.disp].append(r)
                self.req_g[r.g] += r.vol
                add[r.disp] += r.vol
        for (j, r) in self.transit.pop(k, []):
            q[j].append(r)
        self.fc = 0.9 * self.fc + 0.1 * add
        for i in range(n):
            if q[i]:
                keep = []
                for r in q[i]:
                    if t >= r.pat:
                        r.gone = r.pat
                    else:
                        keep.append(r)
                q[i] = keep
        self.queues = q
        up = self.sc.up[k]
        for i in range(n):
            self.cand[i] = self.select(q[i], k, t)
            self.can_serve[i] = bool(up[i]) and self.L[i] >= 1.0

    def run(self, controller):
        rewards = []
        done = False
        while not done:
            self.begin_step()
            r, done = self.decide_and_execute(controller)
            rewards.append(r)
        return rewards

    # ---- metrics
    def metrics(self):
        P = self.P
        tend = H * DT
        reqs = self.allr
        M = len(reqs)
        vol = np.array([r.vol for r in reqs])
        srv = np.array([r.served for r in reqs])
        g = np.array([r.g for r in reqs])
        comp = np.array([r.done is not None for r in reqs])
        aband = np.array([(r.gone is not None) and (r.done is None) for r in reqs])
        exit_t = np.array([(r.first if r.first is not None else (r.gone if r.gone is not None else tend)) for r in reqs])
        wait = exit_t - np.array([r.t for r in reqs])
        wait = np.maximum(wait, 0)
        started = np.array([r.first is not None for r in reqs])
        route = np.array([r.au_route for r in reqs])
        out = {}
        out["n_req"] = M
        out["delivered_ratio"] = srv.sum() / vol.sum()
        out["completed_rate"] = comp.mean()
        out["abandon_rate"] = aband.mean()
        out["unserved_rate"] = (~started).mean()
        out["wait_mean"] = wait.mean()
        out["wait_p95"] = np.percentile(wait, 95)
        out["wait_mean_started"] = wait[started].mean() if started.any() else np.nan
        out["frac_over_tprot"] = (wait >= T_PROT).mean()
        for gi, gn in enumerate(GROUPS):
            m = g == gi
            out[f"wait_{gn}"] = wait[m].mean() if m.any() else np.nan
            out[f"p95_{gn}"] = np.percentile(wait[m], 95) if m.any() else np.nan
            out[f"x_{gn}"] = srv[m].sum() / vol[m].sum() if vol[m].sum() > 0 else np.nan
            out[f"n_{gn}"] = int(m.sum())
        v = g > 0
        out["wait_vuln"] = wait[v].mean() if v.any() else np.nan
        out["x_vuln"] = srv[v].sum() / vol[v].sum()
        xs = np.array([out[f"x_{gn}"] for gn in GROUPS])
        xs = xs[~np.isnan(xs)]
        out["jain"] = float((xs.sum() ** 2) / (len(xs) * (xs ** 2).sum())) if (xs ** 2).sum() > 0 else np.nan
        out["n_comp"] = self.n_comp
        out["n_viol"] = self.n_viol
        out["pc"] = 1 - self.n_viol / self.n_comp if self.n_comp > 0 else np.nan
        out["protq_viol"] = self.n_protviol
        # unauthorised priority (eventual effective priority above true eligibility)
        if self.regime == "claim":
            eb = np.array([r.claim_b for r in reqs])
        elif self.regime == "auth":
            eb = np.array([r.au_b for r in reqs])
        else:
            eb = np.array([r.bt for r in reqs])
        bt = np.array([r.bt for r in reqs])
        out["unauth_rate"] = float((eb > bt + 1e-9).mean())
        out["unauth_n"] = int((eb > bt + 1e-9).sum())
        # vulnerable users that never obtain correct priority
        out["vuln_missed_priority"] = float(((eb < bt - 1e-9) & v).sum() / max(1, v.sum())) if self.regime != "fcfs" else float(1.0)
        # assistance pathway (auth regime only)
        if self.regime == "auth":
            am = route == 3
            out["assist_n"] = int(am.sum())
            out["assist_delay_min"] = float(np.mean([r.au_delay * DT for r in reqs if r.au_route == 3])) if am.any() else 0.0
            out["assist_completion"] = float(comp[am].mean()) if am.any() else np.nan
            for rn, rid in (("token", 0), ("face", 1), ("face_fb", 2), ("assist", 3), ("none", 4), ("misuse", 5)):
                m = route == rid
                out[f"wait_route_{rn}"] = float(wait[m].mean()) if m.any() else np.nan
        else:
            out["assist_n"] = 0; out["assist_delay_min"] = 0.0; out["assist_completion"] = np.nan
        out["offers"] = self.offers
        out["offers_ok"] = self.offers_ok
        out["offer_accept_rate"] = self.offers_ok / self.offers if self.offers else np.nan
        out["hs_fail"] = self.hs_fail
        out["nack"] = self.nack
        out["declined"] = self.declined
        rr = np.array([r.refs > 0 for r in reqs])
        out["referred_completed"] = float(comp[rr].mean()) if rr.any() else np.nan
        out["override"] = self.override
        out["bytes_per_disp_hour"] = self.bytes / (P.n * (H * DT / 60.0))
        out["mean_age"] = self.age_sum / max(1, self.age_n)
        out["ms_per_decision_step"] = 1000.0 * self.compute_s / max(1, self.decisions)
        out["unmet_stock_L"] = float(vol.sum() - srv.sum())
        return out


# -------------------------------------------------------------- controllers
class Heur:
    """Deterministic scheduler: queue manager (via Sim.select) + rule-based referral."""
    timed = True
    uses_centre = False

    def __init__(self, thq=6, thv=40.0, thqj=1, use_ref=True):
        self.thq, self.thv, self.thqj, self.use_ref = thq, thv, thqj, use_ref

    def act(self, sim: Sim):
        n = sim.n
        k_ = sim.k
        acts = [A_WAIT] * n
        for i in range(n):
            c = sim.cand[i]
            if c is None:
                continue
            can = sim.can_serve[i]
            a = -1
            if self.use_ref and c.refs < MAX_REFS and k_ >= c.cool:
                if not can:
                    a = sim.best_neighbour(i, c, False, self.thqj)
                elif len(sim.queues[i]) >= self.thq and c.rem >= self.thv:
                    a = sim.best_neighbour(i, c, True, self.thqj)
            acts[i] = 2 + a if a >= 0 else (A_SERVE if can else A_WAIT)
        return acts


class Central:
    """Centralised optimisation-based scheduler (B5): a central node receives (lossy) reports
    from all dispensers, solves a min-cost assignment of overloaded / blocked dispensers to
    destination slots among the SAME neighbour sets, and sends (lossy) commands.  When a
    command is lost the dispenser applies the local deterministic scheduler without referral."""
    timed = True
    uses_centre = True

    def __init__(self, thq=6, thv=40.0, thqj=1):
        self.thq, self.thv, self.thqj = thq, thv, thqj
        self.cmd_bytes = 0.0

    def act(self, sim: Sim):
        n, k = sim.n, sim.k
        sc = sim.sc
        self.cmd_bytes = 0.0
        fresh = (k - sim.c_step) <= STALE_MAX
        origins = []
        for i in range(n):
            if not fresh[i] or sim.c_has[i] < 1:
                continue
            blocked = (sim.c_up[i] < 1) or sim.c_L[i] < 1.0
            overl = sim.c_Q[i] >= self.thq and sim.c_rem[i] >= self.thv
            if blocked or overl:
                origins.append((i, blocked))
        plan = {}
        if origins:
            slots, cost_cols = [], []
            for j in range(n):
                if not fresh[j] or sim.c_up[j] < 1:
                    continue
                cap = max(0, self.thqj + 1 - int(sim.c_Q[j]))
                for s in range(min(cap, 3)):
                    slots.append((j, s))
            if slots:
                C = np.full((len(origins), len(slots)), 1e6)
                for oi, (i, blk) in enumerate(origins):
                    for si, (j, s) in enumerate(slots):
                        if j in sim.nb[i]:
                            a = int(np.where(sim.nb[i] == j)[0][0])
                            if sim.c_L[j] >= min(sim.c_rem[i], VMAX) and not (sim.cand[i] is not None and sim.cand[i].prev == j):
                                C[oi, si] = sim.c_Q[j] + s + 0.25 * sim.travel[i, a] - (5.0 if blk else 0.0)
                ri, ci = linear_sum_assignment(C)
                for r_, c_ in zip(ri, ci):
                    if C[r_, c_] < 1e5:
                        i = origins[r_][0]
                        j = slots[c_][0]
                        plan[i] = int(np.where(sim.nb[i] == j)[0][0])
        acts = [A_WAIT] * n
        for i in range(n):
            c = sim.cand[i]
            if c is None:
                continue
            can = sim.can_serve[i]
            a = -1
            if i in plan and c.refs < MAX_REFS and k >= c.cool:
                self.cmd_bytes += CMD_BYTES
                if not (sc.loss_c[k, i, 1] or sc.blocked(k, i, -1)):
                    a = plan[i]
            acts[i] = 2 + a if a >= 0 else (A_SERVE if can else A_WAIT)
        return acts


def run_one(seed: int, P: Params, regime: str, controller, rew_w=(1.0, 0.6, 0.3, 0.3, 0.2), sc=None):
    sc = sc or Scenario(seed, P)
    sim = Sim(sc, regime, rew_w)
    sim.run(controller)
    m = sim.metrics()
    m["seed"] = seed
    return m


if __name__ == "__main__":
    import sys
    P = Params()
    t0 = time.time()
    for reg, ctl in (("fcfs", Heur()), ("claim", Heur()), ("auth", Heur()), ("auth", Central()), ("oracle", Heur())):
        m = run_one(1001, P, reg, ctl)
        print(reg, type(ctl).__name__, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in
              ("n_req", "delivered_ratio", "completed_rate", "wait_mean", "wait_p95", "pc", "jain", "offers", "offers_ok")})
    print("elapsed", time.time() - t0)
