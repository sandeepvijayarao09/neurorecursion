"""Evaluate the agentic adaptation layer on REAL cross-day drift: FALCON H1.

FALCON H1 (DANDI 000954): a human participant with intracortical arrays doing
a 7-DoF reach-and-grasp task. "held-in" sessions (Jan 1-20) train the decoder once;
"held-out" sessions (Jan 26 - Feb 9, 6-20 days later) test it.

Per held-out day:
  * first 30% of bins  -> unlabeled adaptation data (agents fit on 60% of it, verifier scores 40%)
  * last 70% of bins   -> evaluation with ground-truth velocity (eval_mask bins only)

Conditions
  frozen         day-0 decoder, no adaptation
  agentic        channel-health + input-gain + latent-alignment agents, label-free verifier
  agentic-noverif same agents, every proposal accepted (ablation: is the verifier needed?)
  single agents  each agent alone, with verifier (ablation)
  recal-fewshot  head refit with labels from the first 30% of the day pooled with held-in data
                 (what the user would have to do today)

The intent agent is disabled here: H1 has no target positions to build pseudo-labels from.
Label-free verifier for H1: decoded-velocity distribution on the verify slice should match
the held-in distribution (symmetric Gaussian KL; lower is better).

Usage: python eval_falcon_h1.py <falcon_h1_dir> <out_dir>
"""
import copy
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import stage_b_spikes_to_cursor as sb  # noqa: E402
from agents import AdaptState, AlignmentAgent, ChannelHealthAgent, FrozenMaster  # noqa: E402

BIN = 0.02
ADAPT_FRAC = 0.3


def load(path):
    with h5py.File(path, "r") as f:
        k = f["acquisition/OpenLoopKinematicsVelocity"]
        V = k["data"][:].astype(np.float32)
        t0 = float(k["starting_time"][()])
        rate = float(k["starting_time"].attrs.get("rate", 1 / BIN))
        if rate < 1:            # FALCON stores the bin period (0.02 s) in this attribute
            rate = 1 / rate
        mask = f["acquisition/eval_mask/data"][:].astype(bool)
        st = f["units/spike_times"][:]
        idx = f["units/spike_times_index"][:]
    edges = t0 + np.arange(len(V) + 1) / rate
    X = np.stack([np.histogram(u, edges)[0] for u in np.split(st, idx[:-1])], 1).astype(np.float32)
    return X, V, mask


def by_day(files):
    days = {}
    for f in sorted(files):
        days.setdefault(os.path.basename(f)[:8], []).append(f)
    out = []
    for d, fs in sorted(days.items()):
        parts = [load(f) for f in fs]
        out.append((d, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                    np.concatenate([p[2] for p in parts])))
    return out


def gauss_kl(A, B):
    """Symmetric KL between Gaussian fits of two sets of decoded velocities."""
    def kl(m0, S0, m1, S1):
        S1i = np.linalg.inv(S1)
        d = m1 - m0
        return 0.5 * (np.trace(S1i @ S0) + d @ S1i @ d - len(m0) + np.log(np.linalg.det(S1) / np.linalg.det(S0)))
    eps = 1e-3 * np.eye(A.shape[1])
    mA, SA = A.mean(0), np.cov(A.T) + eps
    mB, SB = B.mean(0), np.cov(B.T) + eps
    return float(kl(mA, SA, mB, SB) + kl(mB, SB, mA, SA))


class Ctx:   # minimal context the channel/alignment agents need
    def __init__(self, X):
        self.X = X


def r2_masked(V, Vd, m):
    return sb.r2(V[m], Vd[m])


def main(root, out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    held_in = by_day(glob.glob(os.path.join(root, "held_in", "*.nwb")))
    held_out = by_day(glob.glob(os.path.join(root, "held_out", "*.nwb")))
    Xi = np.concatenate([d[1] for d in held_in])
    Vi = np.concatenate([d[2] for d in held_in])
    Mi = np.concatenate([d[3] for d in held_in])
    print(f"held-in: {len(held_in)} days, {len(Xi) * BIN / 60:.1f} min, {Xi.shape[1]} channels, {Vi.shape[1]} DoF")
    print(f"held-out: {len(held_out)} days, {sum(len(d[1]) for d in held_out) * BIN / 60:.1f} min")

    # ---------------- train the master model once (held-in only), then freeze
    print("pretraining encoder on held-in spikes…", flush=True)
    ck = os.path.join(out_dir, "encoder_heldin.pt")
    if os.environ.get("MASTER", "smooth") == "smooth":
        enc = None
    elif os.path.exists(ck):
        enc = sb.SpikeEncoder(Xi.shape[1]).to(sb.DEV)
        enc.load_state_dict(sb.torch.load(ck, map_location=sb.DEV))
    else:
        enc, hist, base = sb.pretrain(Xi, Xi.shape[1], epochs=8, batch=64)
        sb.torch.save(enc.state_dict(), ck)
    embed = lambda A: sb.embed_all(enc, A.astype(np.float32))  # noqa: E731
    hist_fn, LAG = sb.history, sb.LAG_BINS
    if os.environ.get("MASTER", "smooth") == "smooth":
        # classic iBCI decoder features: causally smoothed rates (tau 300 ms), lag 120 ms
        from scipy.signal import lfilter
        a_ = np.exp(-BIN / 0.3)
        embed = lambda A: lfilter([1 - a_], [1, -a_], A.astype(np.float32), axis=0).astype(np.float32)  # noqa: E731
        hist_fn, LAG = (lambda Z: Z), 6
    Ei = embed(Xi)
    Zi = hist_fn(Ei)
    Vli = np.roll(Vi, -LAG, 0)
    tr = np.flatnonzero(Mi)
    k = int(len(tr) * 0.85)
    head = sb.best_ridge(Zi[tr[:k]], Vli[tr[:k]], Zi[tr[k:]], Vli[tr[k:]])
    master = FrozenMaster(embed, head, hist_fn, LAG)
    s0 = AdaptState(Xi.shape[1], Ei.shape[1], n_out=Vi.shape[1])
    ref_out = master.decode(Xi, s0)[Mi]
    held_in_r2 = r2_masked(Vi[tr[k:]], master.decode(Xi, s0)[tr[k:]], np.ones(len(tr) - k, bool))
    ref_rate = Xi.mean(0)
    ref_mu, ref_cov = Ei.mean(0), np.cov(Ei.T) + 1e-3 * np.eye(Ei.shape[1])
    print(f"held-in validation R² {held_in_r2:.3f}", flush=True)

    # Ridge on raw spikes too, as a classic-decoder reference
    Zr = sb.history(Xi)
    raw_head = sb.best_ridge(Zr[tr[:k]], Vli[tr[:k]], Zr[tr[k:]], Vli[tr[k:]])

    def verifier(Xv, state):
        return gauss_kl(master.decode(Xv, state), ref_out)

    def run_agents(Xa, agents_on, use_verifier=True, margin=0.02, loops=2):
        n = len(Xa)
        Xf, Xv = Xa[: int(n * 0.6)], Xa[int(n * 0.6):]
        state = AdaptState(Xi.shape[1], Ei.shape[1], n_out=Vi.shape[1])
        cur = verifier(Xv, state)
        log = []
        ag = {"channel_health": ChannelHealthAgent(ref_rate, 0.2, 3.0, 0.25),
              "alignment": AlignmentAgent(ref_rate, ref_mu, ref_cov)}
        done = set()                        # each agent applies at most once (as in recursive.py)
        for loop in range(loops):
            acc = False
            for name in agents_on:
                if name in done:
                    continue
                a, mode = (name.split("/") + [None])[:2]
                if a == "channel_health":
                    new, info = ag[a].propose(Ctx(Xf), state)
                else:
                    new, info = ag[a].propose(Ctx(Xf), state, master, mode=mode)
                sc = verifier(Xv, new)
                ok = (sc < cur * (1 - margin)) if use_verifier else True
                log.append({"loop": loop, "agent": name, "kl_before": cur, "kl_after": sc, "accepted": bool(ok)})
                if ok and (use_verifier or loop == 0):
                    state, cur, acc = new, sc, True
                    done.add(name)
            if not acc or not use_verifier:
                break
        return state, log

    ALL = ["channel_health", "alignment/input", "alignment/latent"]
    conds = {"agentic": (ALL, True), "agentic_noverifier": (ALL, False),
             "only_channel_health": (["channel_health"], True),
             "only_input_gain": (["alignment/input"], True),
             "only_latent_align": (["alignment/latent"], True)}
    rows = []
    d0 = np.datetime64(f"{held_in[-1][0][:4]}-{held_in[-1][0][4:6]}-{held_in[-1][0][6:8]}")
    for day, X, V, M in held_out:
        n = len(X)
        cut = int(n * ADAPT_FRAC)
        ev = np.zeros(n, bool)
        ev[cut:] = True
        ev &= M
        lag_days = int((np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:8]}") - d0) / np.timedelta64(1, "D"))
        r = {"day": day, "days_after_training": lag_days, "eval_bins": int(ev.sum())}
        r["frozen"] = r2_masked(V, master.decode(X, s0), ev)
        r["raw_ridge_frozen"] = r2_masked(V, np.roll(raw_head(sb.history(X)), sb.LAG_BINS, 0), ev)
        for name, (agents_on, use_v) in conds.items():
            st, lg = run_agents(X[:cut], agents_on, use_v)
            r[name] = r2_masked(V, master.decode(X, st), ev)
            if name == "agentic":
                r["agentic_log"] = lg
                r["agentic_state"] = st.summary()
        # few-shot recalibration with labels from the adaptation slice (the burden we want to remove)
        Zd = hist_fn(embed(X))
        Vld = np.roll(V, -LAG, 0)
        lab = np.flatnonzero(M[:cut])
        h2 = sb.best_ridge(np.concatenate([Zi[tr], Zd[lab[: int(len(lab) * .8)]]]),
                           np.concatenate([Vli[tr], Vld[lab[: int(len(lab) * .8)]]]),
                           Zd[lab[int(len(lab) * .8):]], Vld[lab[int(len(lab) * .8):]])
        r["recal_fewshot"] = r2_masked(V, np.roll(h2(Zd), LAG, 0), ev)
        rows.append(r)
        print(f"{day} (+{lag_days:2d} d): frozen {r['frozen']:.3f} | agentic {r['agentic']:.3f} | "
              f"no-verifier {r['agentic_noverifier']:.3f} | recal {r['recal_fewshot']:.3f} | "
              f"accepted {[l['agent'] for l in r['agentic_log'] if l['accepted']]}", flush=True)

    keys = ["frozen", "raw_ridge_frozen", "agentic", "agentic_noverifier", "only_channel_health",
            "only_input_gain", "only_latent_align", "recal_fewshot"]
    rng = np.random.default_rng(0)
    summ = {}
    for kk in keys:
        v = np.array([r[kk] for r in rows])
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        summ[kk] = {"mean_r2": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    diff = np.array([r["agentic"] - r["frozen"] for r in rows])
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    summ["agentic_minus_frozen"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                    "days_improved": int((diff > 0).sum()), "days": len(diff)}
    gap = np.array([(r["agentic"] - r["frozen"]) / max(1e-6, r["recal_fewshot"] - r["frozen"]) for r in rows
                    if r["recal_fewshot"] > r["frozen"]])
    summ["fraction_of_recal_gain_recovered_median"] = float(np.median(gap)) if len(gap) else None
    out = {"dataset": "FALCON H1 (DANDI 000954), human 7-DoF iBCI", "held_in_val_r2": held_in_r2,
           "summary": summ, "per_day": rows, "runtime_s": round(time.time() - t0, 1),
           "notes": ["intent agent disabled (no targets in H1)", "LLM planner/reviewer not used here (no CRUSOE_API_KEY); rule-based orchestration",
                     "labels used only for evaluation, except the recal_fewshot baseline"]}
    out["master"] = os.environ.get("MASTER", "smooth")
    json.dump(out, open(os.path.join(out_dir, f"falcon_h1_eval_{out['master']}.json"), "w"), indent=2, default=float)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
