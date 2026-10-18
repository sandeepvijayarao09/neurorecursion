"""14-session chronological eval on FALCON H1 (each held-out recording session = one 'day').

Sessions 1-7 (Jan 26 - Feb 2) are the DEV split used to decide which memory-layer changes to keep.
Sessions 8-14 (Feb 3 - Feb 9) are TEST, reported but never used for decisions.
Each variant gets a fresh memory and processes all 14 sessions in order, carrying memory forward.
Labels score the last 70% of each session only; the first 30% is the unlabeled adaptation slice.

Usage: python eval_14_sessions.py <falcon_h1_dir> <out_dir> [variants]
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
from brain_v2 import MemoryOrchestratorV2, MemoryV2  # noqa: E402
from eval_falcon_h1 import BIN, by_day, gauss_kl, load  # noqa: E402

ADAPT_FRAC, LAG, TAU = 0.3, 6, 0.3
VARIANTS = {
    "v1_memory":        dict(cls="v1"),
    "v2_split_half":    dict(cls="v2", split_half=True, delayed_credit=False, gain_bounds=False),
    "v2_delayed_credit": dict(cls="v2", split_half=False, delayed_credit=True, gain_bounds=False),
    "v2_gain_bounds":   dict(cls="v2", split_half=False, delayed_credit=False, gain_bounds=True),
    "v2_all":           dict(cls="v2", split_half=True, delayed_credit=True, gain_bounds=True),
    # round 2 (chosen from round-1 dev results: keep split_half, drop gain_bounds)
    "v3_split_nowarm":   dict(cls="v2", split_half=True, delayed_credit=False, gain_bounds=False, warm_start=False),
    "v3_split_warmsplit": dict(cls="v2", split_half=True, delayed_credit=False, gain_bounds=False, warm_split=True),
    "v3_split_delay_warmsplit": dict(cls="v2", split_half=True, delayed_credit=True, gain_bounds=False, warm_split=True),
}


def setup(root):
    held_in = by_day(glob.glob(os.path.join(root, "held_in", "*.nwb")))
    Xi = np.concatenate([d[1] for d in held_in]); Vi = np.concatenate([d[2] for d in held_in])
    Mi = np.concatenate([d[3] for d in held_in])
    a_ = np.exp(-BIN / TAU)
    embed = lambda A: lfilter([1 - a_], [1, -a_], A.astype(np.float32), axis=0).astype(np.float32)  # noqa: E731
    Ei = embed(Xi); Vli = np.roll(Vi, -LAG, 0)
    tr = np.flatnonzero(Mi); k = int(len(tr) * 0.85)
    head = sb.best_ridge(Ei[tr[:k]], Vli[tr[:k]], Ei[tr[k:]], Vli[tr[k:]])
    master = FrozenMaster(embed, head, lambda Z: Z, LAG)
    tmpl = AdaptState(Xi.shape[1], Ei.shape[1], n_out=Vi.shape[1])
    ref_out = master.decode(Xi, tmpl)[Mi]
    refs = (Xi.mean(0), Ei.mean(0), np.cov(Ei.T) + 1e-3 * np.eye(Ei.shape[1]))
    ctx = dict(Xi=Xi, Vi=Vi, Mi=Mi, Ei=Ei, Vli=Vli, tr=tr, embed=embed)
    return master, tmpl, (lambda Xv, s: gauss_kl(master.decode(Xv, s), ref_out)), refs, ctx


def main(root, out_dir, names=None):
    os.makedirs(out_dir, exist_ok=True)
    master, tmpl, verifier, (ref_rate, ref_mu, ref_cov), c = setup(root)
    files = sorted(glob.glob(os.path.join(root, "held_out", "*.nwb")))
    sessions = [(os.path.basename(f)[:15], *load(f)) for f in files]
    print(f"{len(sessions)} held-out sessions, {sum(len(s[1]) for s in sessions) * BIN / 60:.1f} min")

    def evmask(n, M):
        cut = int(n * ADAPT_FRAC); ev = np.zeros(n, bool); ev[cut:] = True
        return cut, ev & M

    # baselines per session (no memory involved)
    base = []
    for sid, X, V, M in sessions:
        cut, ev = evmask(len(X), M)
        r = {"session": sid, "frozen": sb.r2(V[ev], master.decode(X, tmpl)[ev])}
        Xf, Xv = X[:cut][: int(cut * 0.6)], X[:cut][int(cut * 0.6):]
        st, cur = tmpl, verifier(X[:cut][int(cut * 0.6):], tmpl)
        ch, al = ChannelHealthAgent(ref_rate, 0.2, 3.0, 0.25), AlignmentAgent(ref_rate, ref_mu, ref_cov)
        for nm in ["channel_health", "alignment/input", "alignment/latent"]:
            new, _ = ch.propose(Ctx(Xf), st) if nm == "channel_health" else al.propose(Ctx(Xf), st, master, mode=nm.split("/")[1])
            sc = verifier(Xv, new)
            if sc < cur * 0.98:
                st, cur = new, sc
        r["stateless"] = sb.r2(V[ev], master.decode(X, st)[ev])
        Zd = c["embed"](X); Vld = np.roll(V, -LAG, 0); lab = np.flatnonzero(M[:cut])
        h2 = sb.best_ridge(np.concatenate([c["Ei"][c["tr"]], Zd[lab[: int(len(lab) * .8)]]]),
                           np.concatenate([c["Vli"][c["tr"]], Vld[lab[: int(len(lab) * .8)]]]),
                           Zd[lab[int(len(lab) * .8):]], Vld[lab[int(len(lab) * .8):]])
        r["recal_fewshot"] = sb.r2(V[ev], np.roll(h2(Zd), LAG, 0)[ev])
        base.append(r)

    results = {}
    for name in (names or VARIANTS):
        cfg = dict(VARIANTS[name]); cls = cfg.pop("cls")
        dbp = os.path.join(out_dir, f"memory_{name}.sqlite")
        if os.path.exists(dbp):
            os.remove(dbp)
        mem = MemoryV2(dbp, "HumanPitt-H1")
        Orch = MemoryOrchestrator if cls == "v1" else MemoryOrchestratorV2
        orch = Orch(master, verifier, mem, ref_rate, ref_mu, ref_cov, tmpl, seed=int(os.environ.get("SEED", 0)), **cfg)
        rows = []
        for (sid, X, V, M), b in zip(sessions, base):
            cut, ev = evmask(len(X), M)
            st, info = orch.run_day(sid, X[:cut])
            rows.append({**b, "agentic": sb.r2(V[ev], master.decode(X, st)[ev]), "accepted": info["accepted"],
                         "pruned": info["pruned"], "start": info["start"], "credit": info.get("credit", [])})
        results[name] = {"rows": rows, "memory": mem.summary(len(ref_rate))}
        dev, test = rows[:7], rows[7:]
        f = lambda rs, k: float(np.mean([r[k] for r in rs]))  # noqa: E731
        results[name]["score"] = {"dev": f(dev, "agentic"), "test": f(test, "agentic"), "all": f(rows, "agentic"),
                                  "worst": float(min(r["agentic"] for r in rows)),
                                  "sessions_beating_frozen": int(sum(r["agentic"] > r["frozen"] for r in rows))}
        s = results[name]["score"]
        print(f"{name:18s} dev {s['dev']:+.3f} | test {s['test']:+.3f} | all {s['all']:+.3f} | worst {s['worst']:+.3f} "
              f"| beats frozen {s['sessions_beating_frozen']}/14", flush=True)

    f = lambda k, sl: float(np.mean([r[k] for r in base[sl]]))  # noqa: E731
    results["baselines"] = {k: {"dev": f(k, slice(0, 7)), "test": f(k, slice(7, 14)), "all": f(k, slice(0, 14)),
                                "worst": float(min(r[k] for r in base))} for k in ["frozen", "stateless", "recal_fewshot"]}
    for k, v in results["baselines"].items():
        print(f"{k:18s} dev {v['dev']:+.3f} | test {v['test']:+.3f} | all {v['all']:+.3f} | worst {v['worst']:+.3f}")
    json.dump(results, open(os.path.join(out_dir, os.environ.get("OUT_NAME", "eval_14_sessions.json")), "w"), indent=2, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None)
