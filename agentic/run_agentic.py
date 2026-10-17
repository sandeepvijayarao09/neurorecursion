"""Run the agentic adaptation framework on MC_Maze with simulated day-to-day drift.

Day 0: train the frozen master model once (pretrained encoder + velocity head).
Days 1-4: held-out trials with increasing drift (dead channels, noisy channels,
per-channel gain changes). Compare, on each day's evaluation trials:
  frozen     - day-0 model, no adaptation
  agentic    - frozen model + agent-accepted adaptation state (no labels)
  recal      - head refit daily with ground-truth labels (the calibration we want to avoid)

Usage: python run_agentic.py <mc_maze.nwb> <out_dir>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import stage_b_spikes_to_cursor as sb  # noqa: E402
from agents import (AdaptState, AlignmentAgent, ChannelHealthAgent, DayContext,  # noqa: E402
                    FrozenMaster, IntentAgent, Orchestrator, Verifier, pseudo_labels)

DAYS, TRIALS_PER_DAY, DAY0_TRIALS = 4, 25, 150


def drift(X, sev, rng, perm, gains):
    """Simulated nonstationarity, cumulative with severity."""
    if os.environ.get("NO_DRIFT"):
        return X.copy()
    Y = rng.poisson(X * np.exp(gains * sev)[None, :]).astype(np.float32)
    Y[:, perm[: 6 * sev]] = 0                                     # dead channels
    noisy = perm[60: 60 + 3 * sev]
    Y[:, noisy] += rng.poisson(0.4, (len(Y), len(noisy)))          # noisy channels
    return Y


def main(nwb, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "audit_log.jsonl")
    open(log, "w").close()
    rng = np.random.default_rng(0)

    units, ht, hv, hp, trials = sb.load_nwb(nwb)
    tbin, X, V, P = sb.bin_data(units, ht, hv, hp)
    Vl = np.roll(V, -sb.LAG_BINS, 0)
    st, sp, go = trials["start_time"], trials["stop_time"], trials["go_cue_time"]
    tp, act = trials["target_pos_list"], trials["active_target"].astype(int)
    tgt = np.array([tp[i][act[i]] for i in range(len(st))])
    cands = np.unique(np.round(tgt, -1), axis=0)
    in_trial = np.zeros(len(tbin), bool)
    for s, e in zip(st, sp):
        in_trial[(tbin >= s) & (tbin < e)] = True

    # ---------------- day 0: train the master model once, then freeze it
    d0 = tbin < st[DAY0_TRIALS]
    print("Day 0: pretraining encoder…", flush=True)
    sb_epochs = 12
    enc, _, _ = sb.pretrain(X[d0], X.shape[1], epochs=sb_epochs)
    embed = lambda A: sb.embed_all(enc, A.astype(np.float32))       # noqa: E731
    E0 = embed(X[d0])
    cut = np.flatnonzero(d0 & in_trial)
    k = int(len(cut) * 0.85)
    Zh = sb.history(E0)
    head = sb.best_ridge(Zh[cut[:k]], Vl[cut[:k]], Zh[cut[k:]], Vl[cut[k:]])
    master = FrozenMaster(embed, head, sb.history, sb.LAG_BINS)

    ref_rate = X[d0].mean(0)
    ref_mu, ref_cov = E0.mean(0), np.cov(E0.T) + 1e-3 * np.eye(E0.shape[1])

    def windows_for(idx, t):
        return [(int(np.searchsorted(t, go[i] + 0.25)), int(np.searchsorted(t, go[i] + 1.25))) for i in idx]

    # reference decoded speed in movement windows on day 0
    s0 = AdaptState(X.shape[1], E0.shape[1])
    V0 = master.decode(X[d0], s0)
    w0 = windows_for(range(DAY0_TRIALS), tbin[d0])
    ref_speed = float(np.median(np.linalg.norm(np.concatenate([V0[a:b] for a, b in w0]), axis=1)))

    agents = {"channel_health": ChannelHealthAgent(ref_rate),
              "alignment": AlignmentAgent(ref_rate, ref_mu, ref_cov),
              "intent": IntentAgent(ref_speed)}
    orch = Orchestrator(master, agents, Verifier(ref_speed), max_loops=3, log_path=log)

    perm = rng.permutation(X.shape[1])
    gains = rng.normal(0, 0.2, X.shape[1])
    state = AdaptState(X.shape[1], E0.shape[1])
    rows = []
    for day in range(1, DAYS + 1):
        ids = np.arange(DAY0_TRIALS + (day - 1) * TRIALS_PER_DAY, DAY0_TRIALS + day * TRIALS_PER_DAY)
        m = (tbin >= st[ids[0]]) & (tbin < sp[ids[-1]])
        t_day, P_day, V_day, Vl_day = tbin[m], P[m], V[m], Vl[m]
        X_day = drift(X[m], day, rng, perm, gains)
        local = np.arange(len(ids))
        fit_t, ver_t, ev_t = local[:10], local[10:17], local[17:]
        win = windows_for(ids, t_day)
        start_pos = np.array([P_day[min(a, len(P_day) - 1)] for a, _ in win])   # cursor at go cue
        d = tgt[ids] - start_pos
        ctx = DayContext(X_day, t_day, win, d / np.linalg.norm(d, axis=1, keepdims=True), fit_t, ver_t)

        print(f"Day {day}: orchestrating agents…", flush=True)
        state, score = orch.run_day(day, ctx, state)

        # ---------------- evaluation (ground truth used only here)
        ev_mask = np.zeros(len(t_day), bool)
        for i in ev_t:
            ev_mask[(t_day >= st[ids[i]]) & (t_day < sp[ids[i]])] = True
        sub = {kk: (v[ids[ev_t]] if isinstance(v, np.ndarray) else [v[j] for j in ids[ev_t]])
               for kk, v in trials.items()}
        res = {"day": day}
        Vf = master.decode(X_day, AdaptState(X.shape[1], E0.shape[1]))
        Va = master.decode(X_day, state)
        # daily recalibration baseline: refit head with true labels on fit+verify trials
        lab = np.zeros(len(t_day), bool)
        for i in np.concatenate([fit_t, ver_t]):
            lab[(t_day >= st[ids[i]]) & (t_day < sp[ids[i]])] = True
        Zd = sb.history(embed(X_day))
        li = np.flatnonzero(lab)
        # pooled recalibration: day-0 labels + today's labelled calibration block
        h2 = sb.best_ridge(np.concatenate([Zh[cut[:k]], Zd[li[: int(len(li) * .8)]]]),
                           np.concatenate([Vl[cut[:k]], Vl_day[li[: int(len(li) * .8)]]]),
                           Zd[li[int(len(li) * .8):]], Vl_day[li[int(len(li) * .8):]])
        Vr = np.roll(h2(Zd), sb.LAG_BINS, 0)
        for name, Vd in [("frozen", Vf), ("agentic", Va), ("recal", Vr)]:
            sel, _, _ = sb.selection_eval(sub, t_day, P_day, Vd, name, cands)
            res[name] = {"r2": sb.r2(V_day[ev_mask], Vd[ev_mask]), "acc": sel["accuracy"], "bps": sel["bps"]}
        res["agent_state"] = state.summary()
        res["verifier"] = score
        rows.append(res)
        print(f"  R² frozen={res['frozen']['r2']:.3f} agentic={res['agentic']['r2']:.3f} "
              f"recal={res['recal']['r2']:.3f} | state {res['agent_state']}", flush=True)

    with open(os.path.join(out_dir, "agentic_results.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    days = [r["day"] for r in rows]
    for name, c in [("frozen", "#6e7781"), ("agentic", "#1a7f37"), ("recal", "#8250df")]:
        ax[0].plot(days, [r[name]["r2"] for r in rows], "o-", color=c, label=name)
        ax[1].plot(days, [r[name]["acc"] for r in rows], "o-", color=c, label=name)
    ax[0].set_title("Velocity R² under growing drift")
    ax[1].set_title("Target-selection accuracy")
    for a in ax:
        a.set_xlabel("day (drift severity)")
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "agentic_days.png"), dpi=130)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
