"""Does memory make the agent team better over time? Chronological eval on FALCON H1.

Held-out days are processed in date order (Jan 26 -> Feb 9), like a real user.
  frozen     day-0 decoder
  stateless  agents start from scratch every day, fixed order (previous system)
  memory     MemoryOrchestrator: warm start / rollback, Thompson-sampled schedule,
             channel reliability history, consolidation after each day
Labels are used only to score; the adaptation slice (first 30% of each day) is unlabeled.

Usage: python eval_memory_h1.py <falcon_h1_dir> <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import stage_b_spikes_to_cursor as sb  # noqa: E402
from agents import AdaptState, AlignmentAgent, ChannelHealthAgent, FrozenMaster  # noqa: E402
from brain import Ctx, MemoryOrchestrator  # noqa: E402
from eval_falcon_h1 import BIN, by_day, gauss_kl  # noqa: E402
from memory import UserMemory  # noqa: E402

ADAPT_FRAC, LAG, TAU = 0.3, 6, 0.3


def main(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    held_in = by_day(glob.glob(os.path.join(root, "held_in", "*.nwb")))
    held_out = by_day(glob.glob(os.path.join(root, "held_out", "*.nwb")))
    Xi = np.concatenate([d[1] for d in held_in])
    Vi = np.concatenate([d[2] for d in held_in])
    Mi = np.concatenate([d[3] for d in held_in])
    a_ = np.exp(-BIN / TAU)
    embed = lambda A: lfilter([1 - a_], [1, -a_], A.astype(np.float32), axis=0).astype(np.float32)  # noqa: E731
    Ei = embed(Xi)
    Vli = np.roll(Vi, -LAG, 0)
    tr = np.flatnonzero(Mi)
    k = int(len(tr) * 0.85)
    head = sb.best_ridge(Ei[tr[:k]], Vli[tr[:k]], Ei[tr[k:]], Vli[tr[k:]])
    master = FrozenMaster(embed, head, lambda Z: Z, LAG)
    tmpl = AdaptState(Xi.shape[1], Ei.shape[1], n_out=Vi.shape[1])
    ref_out = master.decode(Xi, tmpl)[Mi]
    ref_rate, ref_mu, ref_cov = Xi.mean(0), Ei.mean(0), np.cov(Ei.T) + 1e-3 * np.eye(Ei.shape[1])
    verifier = lambda Xv, s: gauss_kl(master.decode(Xv, s), ref_out)  # noqa: E731

    def stateless(Xa):
        n = len(Xa)
        Xf, Xv = Xa[: int(n * 0.6)], Xa[int(n * 0.6):]
        st, cur = AdaptState(Xi.shape[1], Ei.shape[1], n_out=Vi.shape[1]), None
        cur = verifier(Xv, st)
        ch = ChannelHealthAgent(ref_rate, 0.2, 3.0, 0.25)
        al = AlignmentAgent(ref_rate, ref_mu, ref_cov)
        for name in ["channel_health", "alignment/input", "alignment/latent"]:
            new, _ = ch.propose(Ctx(Xf), st) if name == "channel_health" else al.propose(Ctx(Xf), st, master, mode=name.split("/")[1])
            sc = verifier(Xv, new)
            if sc < cur * 0.98:
                st, cur = new, sc
        return st

    dbp = os.path.join(out_dir, "user_memory.sqlite")
    if os.path.exists(dbp):
        os.remove(dbp)
    mem = UserMemory(dbp, "HumanPitt-H1")
    orch = MemoryOrchestrator(master, verifier, mem, ref_rate, ref_mu, ref_cov, tmpl)
    rows = []
    for day, X, V, M in held_out:
        n = len(X)
        cut = int(n * ADAPT_FRAC)
        ev = np.zeros(n, bool)
        ev[cut:] = True
        ev &= M
        r2 = lambda s: sb.r2(V[ev], master.decode(X, s)[ev])  # noqa: E731
        s_mem, info = orch.run_day(day, X[:cut])
        row = {"day": day, "frozen": r2(tmpl), "stateless": r2(stateless(X[:cut])), "memory": r2(s_mem), **info}
        rows.append(row)
        print(f"{day}: frozen {row['frozen']:+.3f} | stateless {row['stateless']:+.3f} | memory {row['memory']:+.3f} "
              f"| start={info['start']} order={info['order']} pruned={info['pruned']} accepted={info['accepted']}",
              flush=True)

    rng = np.random.default_rng(0)
    summ = {}
    for kk in ["frozen", "stateless", "memory"]:
        v = np.array([r[kk] for r in rows])
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        summ[kk] = {"mean_r2": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "worst_day": float(v.min())}
    d = np.array([r["memory"] - r["stateless"] for r in rows])
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    summ["memory_minus_stateless"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                      "days_better": int((d > 1e-4).sum()), "days_worse": int((d < -1e-4).sum())}
    summ["memory_contents"] = mem.summary(Xi.shape[1])
    json.dump({"summary": summ, "per_day": rows}, open(os.path.join(out_dir, "memory_eval.json"), "w"), indent=2, default=float)
    print(json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
