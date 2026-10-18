"""7-day evaluation on drift scenarios the agents have never seen.

Conditions, each run on the same 5 unseen scenarios x 7 days (state carried day to day):
  frozen     - day-0 decoder, no adaptation
  agent_v0   - agent layer with the starting playbook
  agent_v1   - agent layer with the playbook learned by the recursive (Nemotron) run
  recal      - daily head refit using real labels (the calibration we want to remove)

Drift grows gradually each day: dead channels, noisy channels, per-channel gain drift.
Ground truth is used only to score. Usage: python seven_day.py <nwb> <out_dir>
"""
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import llm  # noqa: E402
import recursive as rc  # noqa: E402
from agents import AdaptState, DayContext  # noqa: E402

llm.available = lambda: False          # deterministic, no API calls for the eval run
DAYS, PER_DAY, SCENS = 7, 25, [200, 201, 202, 203, 204]


class SevenDayWorld(rc.World):
    """Controlled simulation: decoder trained on a random 60% of trials, so day 0 works;
    each simulated day draws unseen trials and adds drift that grows with the day."""

    def __init__(self, nwb, out_dir):
        import torch
        import stage_b_spikes_to_cursor as sb
        from agents import FrozenMaster
        units, ht, hv, hp, tr = sb.load_nwb(nwb)
        self.tbin, self.X, self.V, self.P = sb.bin_data(units, ht, hv, hp)
        self.tr, self.Vl = tr, np.roll(self.V, -sb.LAG_BINS, 0)
        st, sp = tr["start_time"], tr["stop_time"]
        tp, act = tr["target_pos_list"], tr["active_target"].astype(int)
        self.tgt = np.array([tp[i][act[i]] for i in range(len(st))])
        rng = np.random.default_rng(7)
        order = rng.permutation(len(st))
        self.train_tr, self.pool = np.sort(order[:150]), np.sort(order[150:])
        tr_bins = np.zeros(len(self.tbin), bool)
        for i in self.train_tr:
            tr_bins[(self.tbin >= st[i]) & (self.tbin < sp[i])] = True
        ck = os.path.join(out_dir, "encoder_interleaved.pt")
        if os.path.exists(ck):
            self.enc = sb.SpikeEncoder(self.X.shape[1]).to(sb.DEV)
            self.enc.load_state_dict(torch.load(ck, map_location=sb.DEV))
        else:
            print("pretraining encoder on interleaved training trials…", flush=True)
            self.enc, _, _ = sb.pretrain(self.X[tr_bins], self.X.shape[1], epochs=12)
            torch.save(self.enc.state_dict(), ck)
        embed = lambda A: sb.embed_all(self.enc, A.astype(np.float32))  # noqa: E731
        E = embed(self.X)
        cut = np.flatnonzero(tr_bins)
        k = int(len(cut) * .85)
        Zh = sb.history(E)
        head = sb.best_ridge(Zh[cut[:k]], self.Vl[cut[:k]], Zh[cut[k:]], self.Vl[cut[k:]])
        self.master = FrozenMaster(embed, head, sb.history, sb.LAG_BINS)
        self.Zh_tr, self.Vl_tr = Zh[cut], self.Vl[cut]          # day-0 labelled history (existing system keeps it)
        self.ref_rate = self.X[tr_bins].mean(0)
        E0 = E[tr_bins]
        self.ref_mu, self.ref_cov = E0.mean(0), np.cov(E0.T) + 1e-3 * np.eye(E0.shape[1])
        self.d_lat = E0.shape[1]
        V0 = self.master.decode(self.X, AdaptState(self.X.shape[1], self.d_lat))
        w0 = self._windows(self.train_tr, self.tbin)
        self.ref_speed = float(np.median(np.linalg.norm(np.concatenate([V0[a:b] for a, b in w0 if b > a]), axis=1)))
        self._days = {}

    def day(self, scen, d):
        key = (scen, d)
        if key in self._days:
            return self._days[key]
        rng = np.random.default_rng(1000 + scen)
        perm = rng.permutation(self.X.shape[1])
        gains = rng.normal(0, 0.12, self.X.shape[1])
        rng_d = np.random.default_rng(scen * 10 + d)
        st, sp = self.tr["start_time"], self.tr["stop_time"]
        ids = np.sort(np.random.default_rng(scen * 100 + d).choice(self.pool, PER_DAY, replace=False))
        m = np.zeros(len(self.tbin), bool)
        for i in ids:
            m[(self.tbin >= st[i] - 0.1) & (self.tbin < sp[i])] = True
        g = np.exp(gains * d)[None, :]
        Xm = self.X[m]
        if os.environ.get("NO_DRIFT"):
            Xd = Xm.copy()
        else:   # gain drift without re-drawing every spike: thin when g<1, add extra spikes when g>1
            Xd = (rng_d.binomial(Xm.astype(int), np.minimum(g, 1.0)) + rng_d.poisson(Xm * np.maximum(g - 1, 0))).astype(np.float32)
        Xd[:, perm[: 3 * d]] = 0                                   # 3 more dead channels per day
        noisy = perm[60: 60 + 2 * d]
        Xd[:, noisy] += rng_d.poisson(0.3, (len(Xd), len(noisy)))  # 2 more noisy channels per day
        t_day, P_day = self.tbin[m], self.P[m]
        win = self._windows(ids, t_day)
        start = np.array([P_day[min(a, len(P_day) - 1)] for a, _ in win])
        dv = self.tgt[ids] - start
        local = np.arange(len(ids))
        ctx = DayContext(Xd, t_day, win, dv / np.linalg.norm(dv, axis=1, keepdims=True), local[:8], local[8:13])
        ev = np.zeros(len(t_day), bool)
        for i in local[13:]:
            ev[(t_day >= st[ids[i]]) & (t_day < sp[ids[i]])] = True
        lab = np.zeros(len(t_day), bool)
        for i in local[:13]:
            lab[(t_day >= st[ids[i]]) & (t_day < sp[ids[i]])] = True
        self._days[key] = (ctx, {"V": self.V[m], "Vl": self.Vl[m], "ev": ev, "lab": lab})
        return self._days[key]


def existing_r2(world, scen, d):
    """Existing system: encoder 99% frozen, the 1% user head is retrained every day on
    day-0 labels plus today's calibration labels."""
    import stage_b_spikes_to_cursor as sb
    ctx, truth = world.day(scen, d)
    Z = sb.history(world.master.embed_fn(ctx.X))
    li = np.flatnonzero(truth["lab"])
    k = int(len(li) * .8)
    head = sb.best_ridge(np.concatenate([world.Zh_tr, Z[li[:k]]]), np.concatenate([world.Vl_tr, truth["Vl"][li[:k]]]),
                         Z[li[k:]], truth["Vl"][li[k:]])
    V = np.roll(head(Z), sb.LAG_BINS, 0)
    return sb.r2(truth["V"][truth["ev"]], V[truth["ev"]])


def intent_cos(V, ctx, trials):
    idx, dirs = [], []
    for i in trials:
        a, b = ctx.windows[i]
        idx.append(np.arange(a, b)); dirs.append(np.repeat(ctx.intent_dir[i][None], b - a, 0))
    idx, dirs = np.concatenate(idx), np.concatenate(dirs)
    v = V[idx]
    return float(np.mean((v * dirs).sum(1) / (np.linalg.norm(v, axis=1) + 1e-6)))


def memory_decode(world, scen, d, state, mem):
    """Agent long-term user memory: every day the agents store label-free examples
    (adapted signal -> direction to the known target, at the decoded speed). The memory
    grows across days; a small output translation is fit on all of it and used only
    if the label-free verifier score improves. The foundation model stays frozen."""
    import stage_b_spikes_to_cursor as sb
    ctx, truth = world.day(scen, d)
    Z = world.master.latent(ctx.X, state)
    Z = (Z - state.lat_mu) @ state.lat_A + state.lat_mu0
    H = sb.history(Z)
    V_base = world.master.decode(ctx.X, state)
    for i in ctx.fit_trials:
        a, b = ctx.windows[i]
        idx = np.arange(max(a, sb.LAG_BINS), b)
        sp = np.linalg.norm(V_base[idx], axis=1, keepdims=True)
        mem["X"].append(H[idx - sb.LAG_BINS]); mem["y"].append(ctx.intent_dir[i][None] * sp)
    Xm, ym = np.concatenate(mem["X"]), np.concatenate(mem["y"])
    k = int(len(Xm) * .8)
    f = sb.best_ridge(Xm[:k], ym[:k], Xm[k:], ym[k:])
    V_mem = np.roll(f(H), sb.LAG_BINS, 0)
    V_blend = 0.5 * V_base + 0.5 * V_mem
    use = intent_cos(V_blend, ctx, ctx.verify_trials) > intent_cos(V_base, ctx, ctx.verify_trials) + 0.01
    V = V_blend if use else V_base
    return sb.r2(truth["V"][truth["ev"]], V[truth["ev"]]), use, len(Xm)


def main(nwb, out_dir):
    t0 = time.time()
    world = SevenDayWorld(nwb, os.path.join(os.path.dirname(__file__), "out_recursive"))
    brain = rc.SharedBrain(os.path.join(out_dir, "memory_7day.json"))
    v1 = json.load(open(os.path.join(os.path.dirname(__file__), "out_recursive", "playbook_final.json")))
    books = {"agent": copy.deepcopy(rc.PLAYBOOK_V0)}
    res = {k: np.zeros((len(SCENS), DAYS)) for k in ["frozen", "existing", "agent_no_memory", "agent_memory"]}
    masked = {"agent": np.zeros((len(SCENS), DAYS))}
    used = np.zeros((len(SCENS), DAYS)); mem_size = np.zeros((len(SCENS), DAYS))
    log = lambda r: None
    for si, scen in enumerate(SCENS):
        state = AdaptState(world.X.shape[1], world.d_lat)
        mem = {"X": [], "y": []}
        for d in range(1, DAYS + 1):
            res["frozen"][si, d - 1] = world.r2(scen, d, AdaptState(world.X.shape[1], world.d_lat))
            res["existing"][si, d - 1] = existing_r2(world, scen, d)
            state, _, _ = rc.run_day(world, books["agent"], brain, scen, d, state, log, use_llm=False)
            res["agent_no_memory"][si, d - 1] = world.r2(scen, d, state)
            r, u, n = memory_decode(world, scen, d, state, mem)
            res["agent_memory"][si, d - 1], used[si, d - 1], mem_size[si, d - 1] = r, u, n
            masked["agent"][si, d - 1] = (state.ch_weight == 0).sum()
        print(f"scenario {scen}: " + "  ".join(f"{k} {res[k][si].mean():.3f}" for k in res), flush=True)
    v1 = None
    out = {"days": DAYS, "scenarios": SCENS, "trials_per_day": PER_DAY, "eval_trials_per_day": PER_DAY - 13, "setup": "controlled: decoder trained on random 60% of trials; days sample unseen trials + growing synthetic drift",
           "mean": {k: v.mean(0).tolist() for k, v in res.items()},
           "sem": {k: (v.std(0) / np.sqrt(len(SCENS))).tolist() for k, v in res.items()},
           "per_scenario": {k: v.tolist() for k, v in res.items()},
           "masked_mean": {k: v.mean(0).tolist() for k, v in masked.items()},
           "memory_used_frac": used.mean(0).tolist(), "memory_examples": mem_size.mean(0).tolist(),
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(out_dir, "seven_day_memory.json"), "w"), indent=2)
    for k in res:
        print(k.ljust(9), " ".join(f"{x:6.3f}" for x in out["mean"][k]), f"| mean {np.mean(out['mean'][k]):.3f}")
    print(f"done in {out['runtime_s']}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
