"""v2: label agents with a split-half labelled verifier and an adapter anchored to training data.

Agents that USE calibration labels, vs the current system (FALCON H1, 14 sessions).

Each session gives every labelled method the same budget: labels on the first 30% (the
calibration slice). Scoring is on the last 70%, as before.

  current system      refit the decoder head with the slice's labels (pooled with training data)
  label agents        frozen decoder; agents propose changes, a LABELLED verifier (R² on the
                      held-back 40% of the slice) accepts only real improvements; new agent
                      `output_adapter` fits a small 7x7 correction from labels (the "1%" layer)
  label agents + mem  same, plus memory: start from the last verified state when it scores better
  hybrid              current system's refit head + label agents on top

Dev = sessions 1-7, test = sessions 8-14 (reported only).
Usage: python eval_labeled_agents.py <falcon_h1_dir> <out_dir>
"""
import copy
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import stage_b_spikes_to_cursor as sb  # noqa: E402
from agents import AlignmentAgent, ChannelHealthAgent, FrozenMaster  # noqa: E402
from brain import Ctx  # noqa: E402
from eval_14_sessions import ADAPT_FRAC, LAG, setup  # noqa: E402
from eval_falcon_h1 import load  # noqa: E402

MARGIN = 0.005


ANCHOR = {}


def output_adapter(master, Xf, Vf, Mf, state, lam=0.1):
    """Ridge-fit a 7x7 correction toward the labels, pooled with training-data pairs (anchor)
    so ~10 s of session labels cannot pull it far from what worked on the training days."""
    A = master.decode(Xf, state)[Mf]
    T = Vf[Mf]
    if ANCHOR:
        k = ANCHOR["X"]
        Ai = master.decode(k, state)[ANCHOR["M"]][::ANCHOR["step"]]
        A, T = np.concatenate([A, Ai]), np.concatenate([T, ANCHOR["V"][ANCHOR["M"]][::ANCHOR["step"]]])
    G = A.T @ A
    reg = lam * np.trace(G) / len(G) * np.eye(len(G))
    D = np.eye(len(G)) + np.linalg.solve(G + reg, A.T @ (T - A))
    new = copy.deepcopy(state)
    new.out_M = (state.out_M @ D).astype(np.float32)
    return new, {"change": float(np.abs(D - np.eye(len(D))).mean())}


def run_label_agents(master, tmpl, refs, X, V, M, start=None, master_override=None):
    m = master_override or master
    ref_rate, ref_mu, ref_cov = refs
    n = len(X)
    f = int(n * 0.6)
    Xf, Vf, Mf, Xv, Vv, Mv = X[:f], V[:f], M[:f], X[f:], V[f:], M[f:]
    h = len(Xv) // 2
    def score(s):          # labelled verifier: mean R² over the two halves, plus each half
        P = m.decode(Xv, s)
        a_, b_ = sb.r2(Vv[:h][Mv[:h]], P[:h][Mv[:h]]), sb.r2(Vv[h:][Mv[h:]], P[h:][Mv[h:]])
        return np.array([(a_ + b_) / 2, a_, b_])
    fresh = copy.deepcopy(tmpl)
    state, cur, src = fresh, score(fresh), "fresh"
    if start is not None:
        sc = score(start)
        if (sc[1:] > cur[1:]).all():
            state, cur, src = start, sc, "memory"
    ch, al = ChannelHealthAgent(ref_rate, 0.2, 3.0, 0.25), AlignmentAgent(ref_rate, ref_mu, ref_cov)
    log = []
    for name in ["channel_health", "alignment/input", "alignment/latent", "output_adapter"]:
        if name == "channel_health":
            new, info = ch.propose(Ctx(Xf), state)
        elif name == "output_adapter":
            new, info = output_adapter(m, Xf, Vf, Mf, state)
        else:
            new, info = al.propose(Ctx(Xf), state, m, mode=name.split("/")[1])
        sc = score(new)
        ok = bool((sc[1:] > cur[1:] + MARGIN).all())        # must help on BOTH labelled halves
        log.append({"agent": name, "before": float(cur[0]), "after": float(sc[0]), "accepted": ok})
        if ok:
            state, cur = new, sc
    return state, {"start": src, "accepted": [l["agent"] for l in log if l["accepted"]], "log": log}


def main(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    master, tmpl, _, refs, c = setup(root)
    ANCHOR.update({"X": c["Xi"], "V": c["Vi"], "M": c["Mi"], "step": 20})   # ~6.8k training pairs
    files = sorted(glob.glob(os.path.join(root, "held_out", "*.nwb")))
    lf = json.load(open(os.path.join(os.path.dirname(__file__), "out_14sessions_fixed", "eval_14_fixed.json")))
    labelfree = [r["agentic"] for r in lf["v3_split_delay_warmsplit"]["rows"]]
    rows, mem_state, mem_state_h = [], None, None
    for i, f in enumerate(files):
        sid = os.path.basename(f)[:15]
        X, V, M = load(f)
        n = len(X); cut = int(n * ADAPT_FRAC)
        ev = np.zeros(n, bool); ev[cut:] = True; ev &= M
        r2 = lambda P: sb.r2(V[ev], P[ev])  # noqa: E731
        # current system: refit head with this session's slice labels
        Zd = c["embed"](X); Vld = np.roll(V, -LAG, 0); lab = np.flatnonzero(M[:cut])
        h2 = sb.best_ridge(np.concatenate([c["Ei"][c["tr"]], Zd[lab[: int(len(lab) * .8)]]]),
                           np.concatenate([c["Vli"][c["tr"]], Vld[lab[: int(len(lab) * .8)]]]),
                           Zd[lab[int(len(lab) * .8):]], Vld[lab[int(len(lab) * .8):]])
        current = np.roll(h2(Zd), LAG, 0)
        # label agents, no memory / with memory
        s_a, info_a = run_label_agents(master, tmpl, refs, X[:cut], V[:cut], M[:cut])
        s_m, info_m = run_label_agents(master, tmpl, refs, X[:cut], V[:cut], M[:cut], start=mem_state)
        mem_state = s_m
        # hybrid: current system's refit head, then label agents around it (with memory)
        recal_master = FrozenMaster(c["embed"], h2, lambda Z: Z, LAG)
        s_h, info_h = run_label_agents(master, tmpl, refs, X[:cut], V[:cut], M[:cut], start=mem_state_h,
                                       master_override=recal_master)
        mem_state_h = s_h
        row = {"session": sid, "frozen": r2(master.decode(X, tmpl)), "current": r2(current),
               "label_free_agents": labelfree[i], "label_agents": r2(master.decode(X, s_a)),
               "label_agents_memory": r2(master.decode(X, s_m)), "hybrid": r2(recal_master.decode(X, s_h)),
               "accepted_label_agents": info_a["accepted"], "accepted_memory": info_m["accepted"],
               "memory_start": info_m["start"], "accepted_hybrid": info_h["accepted"]}
        rows.append(row)
        print(f"{sid}: frozen {row['frozen']:+.3f} | current {row['current']:+.3f} | label-free {row['label_free_agents']:+.3f} | "
              f"label agents {row['label_agents']:+.3f} | +memory {row['label_agents_memory']:+.3f} | hybrid {row['hybrid']:+.3f}"
              f" | acc {info_m['accepted']}", flush=True)
    keys = ["frozen", "label_free_agents", "current", "label_agents", "label_agents_memory", "hybrid"]
    summ = {}
    rng = np.random.default_rng(0)
    for k in keys:
        v = np.array([r[k] for r in rows])
        summ[k] = {"dev": float(v[:7].mean()), "test": float(v[7:].mean()), "all": float(v.mean()), "worst": float(v.min()),
                   "beats_current": int(sum(r[k] > r["current"] + 0.005 for r in rows))}
    for k in ["label_agents", "label_agents_memory", "hybrid"]:
        d = np.array([r[k] - r["current"] for r in rows])
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        summ[k]["minus_current"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out_dir, "labeled_agents_v2.json"), "w"), indent=2, default=float)
    for k in keys:
        s = summ[k]
        print(f"{k:22s} dev {s['dev']:+.3f} | test {s['test']:+.3f} | all {s['all']:+.3f} | worst {s['worst']:+.3f} | beats current {s['beats_current']}/14"
              + (f" | vs current {s['minus_current']['mean']:+.3f} CI {s['minus_current']['ci95'][0]:+.3f},{s['minus_current']['ci95'][1]:+.3f}" if "minus_current" in s else ""))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
