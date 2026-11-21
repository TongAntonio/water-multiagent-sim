"""
Parameter-sharing PPO with centralised training / decentralised execution (CTDE).
  * actor  : shared MLP, input = local observation o_i(k) only (OBS_DIM features)
  * critic : centralised value network, input = [own observation, pooled (mean,max) observations of
             all dispensers, time fraction, packet-loss rate]  (simulator state: training only)
  * reward : the team reward of Eq. (16) is decomposed additively over dispensers
             (sum_i r_i = r_k).  For credit assignment each agent is trained on the
             neighbourhood-averaged share  n*(0.5 r_i + 0.5 mean_{j in N_i} r_j)  minus a small
             shaping penalty for supervisor overrides.  All reported metrics are team metrics.
  * the sampled *proposed* action and its log-probability are kept for the update; the
    safety supervisor may replace the executed action (override).
Seeds are disjoint:
    train 1000-4999 | heuristic calibration 5000-5005 | checkpoint selection 6000-6003 | test 9000-9029
Usage:  python ppo.py B4 scratch|bc [updates] [episodes_per_update]   (B4: authenticated eligibility)
        python ppo.py B2 scratch|bc ...                                    (B2: unverified claims)
"""
import sys, time, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from sim import *

torch.set_num_threads(1)
GF_DIM = 2 * OBS_DIM + 2
LAMBDA_OVR = 0.05


class AC(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(OBS_DIM, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, ACT_DIM))
        self.critic = nn.Sequential(nn.Linear(OBS_DIM + GF_DIM, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        with torch.no_grad():                       # prior: serve unless something suggests otherwise
            self.actor[-1].bias.copy_(torch.tensor([1.5] + [0.0] * (ACT_DIM - 1)))
            self.actor[-1].weight.mul_(0.1)

    @staticmethod
    def gfeat(obs, k, rho):
        return torch.cat([obs.mean(0), obs.max(0).values, torch.tensor([k / H, rho], dtype=torch.float32)])


class Learned:
    """Controller wrapper.  greedy=True for evaluation, False for training rollouts."""
    timed = True
    uses_centre = False

    def __init__(self, model, greedy=True, rho=0.0, record=False):
        self.m, self.greedy, self.rho, self.record = model, greedy, rho, record
        self.buf = dict(obs=[], act=[], logp=[], gf=[])

    def act(self, sim):
        x = torch.from_numpy(sim.obs())
        with torch.no_grad():
            logits = self.m.actor(x)
            a = logits.argmax(-1) if self.greedy else Categorical(logits=logits).sample()
        if self.record:
            lp = torch.log_softmax(logits, -1).gather(1, a[:, None])[:, 0]
            self.buf["obs"].append(x); self.buf["act"].append(a); self.buf["logp"].append(lp)
            self.buf["gf"].append(AC.gfeat(x, sim.k, self.rho))
        return a.tolist()


def rollout(model, seed, P, regime, greedy=False, record=True):
    sc = Scenario(seed, P)
    sim = Sim(sc, regime)
    ctl = Learned(model, greedy=greedy, rho=P.rho, record=record)
    team, ar, ov = [], [], []
    done = False
    while not done:
        sim.begin_step()
        r, done = sim.decide_and_execute(ctl)
        team.append(r); ar.append(sim.agent_rew.copy()); ov.append(sim.agent_ovr.copy())
    return sim, ctl, np.array(team, np.float32), np.array(ar, np.float32), np.array(ov, np.float32), sc.nb


class Recorder:
    """Runs the rule-based scheduler while recording (observation, action) pairs for behaviour cloning."""
    timed = False
    uses_centre = False

    def __init__(self, h):
        self.h, self.X, self.Y = h, [], []

    def act(self, sim):
        x = sim.obs()
        a = self.h.act(sim)
        self.X.append(x); self.Y.append(np.array(a))
        return a


def bc_pretrain(model, regime, episodes=150, seed=7):
    """Warm start: imitate the calibrated rule-based referral policy (supervised), PPO then fine-tunes."""
    rng = np.random.default_rng(seed)
    rec = Recorder(Heur(12, 60.0, 2))
    for e in range(episodes):
        s = int(rng.integers(1000, 5000))
        sim = Sim(Scenario(s, Params(rho=float(rng.choice([0.0, 0.2, 0.4])))), regime)
        sim.run(rec)
    X = torch.from_numpy(np.concatenate(rec.X)); Y = torch.from_numpy(np.concatenate(rec.Y)).long()
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    for ep in range(8):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 2048):
            idx = perm[i:i + 2048]
            loss = nn.functional.cross_entropy(model.actor(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
    with torch.no_grad():
        acc = (model.actor(X).argmax(-1) == Y).float().mean().item()
    print("BC accuracy", round(acc, 4), "samples", len(X), flush=True)
    return acc


def evaluate(model, seeds, regime, rhos=(0.0, 0.2, 0.4)):
    out = []
    for rho in rhos:
        for s in seeds:
            sim, ctl, team, *_ = rollout(model, s, Params(rho=rho), regime, greedy=True, record=False)
            m = sim.metrics()
            out.append((team.sum(), m["delivered_ratio"], m["wait_p95"]))
    return np.array(out).mean(0)


def train(name, init="scratch", updates=120, eps=16, seed=0):
    regime = {"B4": "auth", "B2": "claim"}[name]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 123)
    model = AC()
    if init == "bc":
        bc_pretrain(model, regime)
    lr_a = 1e-4 if init == "scratch" else 3e-5
    opt_a = torch.optim.Adam(model.actor.parameters(), lr=lr_a)
    opt_c = torch.optim.Adam(model.critic.parameters(), lr=1e-3)
    gamma, lam, clip = 0.95, 0.9, 0.1
    best, best_state, hist = -1e9, None, []
    r0, dr0, w0 = evaluate(model, range(6000, 6004), regime)
    best, best_state = float(r0), {k_: v_.clone() for k_, v_ in model.state_dict().items()}
    print(name, init, "initial val_return", round(float(r0), 3), "delivered", round(float(dr0), 4), flush=True)
    t0 = time.time()
    for u in range(updates):
        ent_coef = 0.003 * (1 - u / updates) + 0.0003
        OBS, ACT, LP, ADV, CIN, RET = [], [], [], [], [], []
        ep_ret = []
        for e in range(eps):
            s = int(rng.integers(1000, 5000))
            rho = float(rng.choice([0.0, 0.2, 0.4]))
            sim, ctl, team, ar, ov, nb = rollout(model, s, Params(rho=rho), regime, greedy=False, record=True)
            n = ar.shape[1]
            rt = n * (0.7 * ar + 0.3 * ar[:, nb].mean(-1)) - LAMBDA_OVR * ov          # (H, n)
            obs = torch.stack(ctl.buf["obs"]); act = torch.stack(ctl.buf["act"])
            lp = torch.stack(ctl.buf["logp"]); gf = torch.stack(ctl.buf["gf"])           # (H,n,d) (H,n) (H,n) (H,G)
            cin = torch.cat([obs, gf[:, None, :].expand(-1, n, -1)], -1)                  # (H, n, d+G)
            with torch.no_grad():
                v = model.critic(cin)[..., 0].numpy()                                     # (H, n)
            adv = np.zeros((H, n), np.float32)
            last = np.zeros(n, np.float32)
            for k in reversed(range(H)):
                nv = v[k + 1] if k + 1 < H else 0.0
                delta = rt[k] + gamma * nv - v[k]
                last = delta + gamma * lam * last
                adv[k] = last
            ret = adv + v
            OBS.append(obs); ACT.append(act); LP.append(lp); CIN.append(cin)
            ADV.append(torch.from_numpy(adv)); RET.append(torch.from_numpy(ret.astype(np.float32)))
            ep_ret.append(float(team.sum()))
        obs = torch.cat(OBS).reshape(-1, OBS_DIM); act = torch.cat(ACT).reshape(-1); lp_old = torch.cat(LP).reshape(-1)
        adv = torch.cat(ADV).reshape(-1); ret = torch.cat(RET).reshape(-1); cin = torch.cat(CIN).reshape(-1, OBS_DIM + GF_DIM)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.shape[0]
        for ep in range(3):
            perm = torch.randperm(N)
            for i in range(0, N, 4096):
                idx = perm[i:i + 4096]
                dist = Categorical(logits=model.actor(obs[idx]))
                ratio = torch.exp(dist.log_prob(act[idx]) - lp_old[idx])
                a = adv[idx]
                loss_a = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean() - ent_coef * dist.entropy().mean()
                opt_a.zero_grad(); loss_a.backward()
                nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5); opt_a.step()
                loss_c = ((model.critic(cin[idx])[:, 0] - ret[idx]) ** 2).mean()
                opt_c.zero_grad(); loss_c.backward()
                nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5); opt_c.step()
        rec = dict(update=u, mean_return=float(np.mean(ep_ret)), entropy=float(dist.entropy().detach().mean()), t=time.time() - t0)
        if (u + 1) % 10 == 0 or u == updates - 1:
            r, dr, w95 = evaluate(model, range(6000, 6004), regime)
            rec.update(val_return=float(r), val_delivered=float(dr), val_p95=float(w95))
            if r > best:
                best = r
                best_state = {k_: v_.clone() for k_, v_ in model.state_dict().items()}
                rec["best"] = True
            print(name, json.dumps({k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in rec.items()}), flush=True)
        hist.append(rec)
    os.makedirs("models", exist_ok=True)
    torch.save(best_state, f"models/{name}_{init}.pt")
    json.dump(hist, open(f"logs/train_{name}_{init}.json", "w"))
    return best_state


def load(name):
    m = AC()
    m.load_state_dict(torch.load(f"models/{name}.pt"))
    m.eval()
    return m


if __name__ == "__main__":
    name, init = sys.argv[1], sys.argv[2]
    U = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    E = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    train(name, init, U, E)
