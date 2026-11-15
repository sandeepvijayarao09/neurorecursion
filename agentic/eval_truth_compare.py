"""Truth vs agent system vs current system, per session (FALCON H1, 14 held-out sessions).

  truth           recorded 7-DoF velocity
  agent system    frozen decoder + memory v3 agents (no labels on new sessions)
  current system  decoder refit each session with that session's calibration labels (few-shot)

Writes per-session metrics (R², Pearson r, RMSE vs truth) and downsampled traces for plotting.
Usage: python eval_truth_compare.py <falcon_h1_dir> <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import stage_b_spikes_to_cursor as sb  # noqa: E402  (path added by eval_14_sessions)
from brain_v2 import MemoryOrchestratorV2, MemoryV2  # noqa: E402
from eval_14_sessions import ADAPT_FRAC, LAG, setup  # noqa: E402
from eval_falcon_h1 import BIN, load  # noqa: E402

DOF = ["vx", "vy", "vz", "rx", "g1", "g2", "g3"]
STEP = 5   # plot at 10 Hz


def metrics(T, P):
    r2 = sb.r2(T, P)
    r = np.nanmean([np.corrcoef(T[:, i], P[:, i])[0, 1] for i in range(T.shape[1]) if T[:, i].std() > 0])
    rmse = float(np.sqrt(np.mean((T - P) ** 2)))
    return {"r2": float(r2), "corr": float(r), "rmse": rmse}


def main(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    master, tmpl, verifier, (ref_rate, ref_mu, ref_cov), c = setup(root)
    files = sorted(glob.glob(os.path.join(root, "held_out", "*.nwb")))
    dbp = os.path.join(out_dir, "memory_truth_compare.sqlite")
    if os.path.exists(dbp):
        os.remove(dbp)
    orch = MemoryOrchestratorV2(master, verifier, MemoryV2(dbp, "HumanPitt-H1"), ref_rate, ref_mu, ref_cov, tmpl,
                                split_half=True, delayed_credit=True, gain_bounds=False, warm_split=True, seed=0)
    out = {"dof": DOF, "hz": round(1 / (BIN * STEP)), "sessions": []}
    for f in files:
        sid = os.path.basename(f)[:15]
        X, V, M = load(f)
        n = len(X); cut = int(n * ADAPT_FRAC)
        ev = np.zeros(n, bool); ev[cut:] = True; ev &= M
        st, info = orch.run_day(sid, X[:cut])
        Va = master.decode(X, st)
        Zd = c["embed"](X); Vld = np.roll(V, -LAG, 0); lab = np.flatnonzero(M[:cut])
        h2 = sb.best_ridge(np.concatenate([c["Ei"][c["tr"]], Zd[lab[: int(len(lab) * .8)]]]),
                           np.concatenate([c["Vli"][c["tr"]], Vld[lab[: int(len(lab) * .8)]]]),
                           Zd[lab[int(len(lab) * .8):]], Vld[lab[int(len(lab) * .8):]])
        Vc = np.roll(h2(Zd), LAG, 0)
        Vf = master.decode(X, tmpl)
        T, A, C, F = V[ev], Va[ev], Vc[ev], Vf[ev]
        idx = np.arange(0, len(T), STEP)
        sess = {"id": sid, "label": f"{sid[4:6]}/{sid[6:8]} {sid[9:11]}:{sid[11:13]}", "seconds": round(len(T) * BIN, 1),
                "agent": metrics(T, A), "current": metrics(T, C), "frozen": metrics(T, F), "accepted": info["accepted"],
                "trace": {k: np.round(v[idx], 4).T.tolist() for k, v in [("truth", T), ("agent", A), ("current", C)]}}
        out["sessions"].append(sess)
        print(f"{sid}: R² agent {sess['agent']['r2']:+.3f}  current {sess['current']['r2']:+.3f}  frozen {sess['frozen']['r2']:+.3f}"
              f" | r agent {sess['agent']['corr']:.2f} current {sess['current']['corr']:.2f}", flush=True)
    for k in ["agent", "current", "frozen"]:
        out[f"mean_{k}"] = {m: float(np.mean([s[k][m] for s in out["sessions"]])) for m in ["r2", "corr", "rmse"]}
    json.dump(out, open(os.path.join(out_dir, "truth_compare.json"), "w"))
    print(json.dumps({k: out[k] for k in out if k.startswith("mean_")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
