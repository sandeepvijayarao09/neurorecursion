"""Stage B, 5-fold blocked cross-validation.

Re-runs the stage_b pipeline with every block of trials held out in turn, so the
target-selection BPS is scored on all trials instead of the last 20%. The encoder
is re-pretrained in each fold on training-period spikes only.

Usage: python stage_b_crossval.py <mc_maze.nwb> <out_dir>
"""
import json
import os
import sys
import time

import numpy as np

import stage_b_spikes_to_cursor as sb

K = 5
FRACS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]


def subset(trials, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) else [x for x, m in zip(v, keep) if m])
            for k, v in trials.items()}


def per_trial_hits(trials, tbin, P, Vdec, cands):
    """Per-trial correct/incorrect + time, for paired bootstrap."""
    hits, secs = [], []
    for i in range(len(trials["start_time"])):
        one = subset(trials, np.arange(len(trials["start_time"])) == i)
        r, _, _ = sb.selection_eval(one, tbin, P, Vdec, "", cands)
        if r["trials"]:
            hits.append(r["correct"])
            secs.append(r["seconds"])
    return np.array(hits), np.array(secs)


def main(nwb, out_dir):
    t0 = time.time()
    units, ht, hv, hp, trials = sb.load_nwb(nwb)
    tbin, X, V, P = sb.bin_data(units, ht, hv, hp)
    Vl = np.roll(V, -sb.LAG_BINS, 0)
    st, sp = trials["start_time"], trials["stop_time"]
    n = len(st)
    # all candidate targets come from the whole session so every fold uses the same N
    tp, act = trials["target_pos_list"], trials["active_target"].astype(int)
    cands = np.unique(np.round(np.array([tp[i][act[i]] for i in range(n)]), -1), axis=0)
    n_targets = len(cands)

    in_trial = np.zeros(len(tbin), bool)
    trial_of_bin = np.full(len(tbin), -1)
    for i, (s, e) in enumerate(zip(st, sp)):
        m = (tbin >= s) & (tbin < e)
        in_trial[m] = True
        trial_of_bin[m] = i
    moving = np.zeros(len(tbin), bool)
    for on, e in zip(trials["move_onset_time"], sp):
        if not np.isnan(on):
            moving[(tbin >= on) & (tbin < e)] = True

    blocks = np.array_split(np.arange(n), K)
    folds, hits = [], {"true": [], "raw": [], "emb": []}
    secs_all = []
    curves = {f: {"raw": [], "emb": []} for f in FRACS}
    for k, test_tr in enumerate(blocks):
        val_tr = blocks[k - 1][-len(test_tr) // 2:] if k > 0 else blocks[1][: len(test_tr) // 2]
        t_lo, t_hi = st[test_tr[0]], sp[test_tr[-1]]
        v_lo, v_hi = st[val_tr[0]], sp[val_tr[-1]]
        te_m = (tbin >= t_lo) & (tbin < t_hi)
        va_m = (tbin >= v_lo) & (tbin < v_hi)
        tr_m = ~te_m & ~va_m
        print(f"fold {k + 1}/{K}: test trials {test_tr[0]}-{test_tr[-1]}", flush=True)

        enc, hist, base = sb.pretrain(X[tr_m], X.shape[1])
        E = sb.embed_all(enc, X)
        Zraw, Zemb = sb.history(X), sb.history(E)
        tr_idx = np.flatnonzero(tr_m & in_trial)
        va, te = va_m & in_trial, te_m & in_trial
        for frac in FRACS:
            sub = tr_idx[: max(50, int(len(tr_idx) * frac))]
            f_raw = sb.best_ridge(Zraw[sub], Vl[sub], Zraw[va], Vl[va])
            f_emb = sb.best_ridge(Zemb[sub], Vl[sub], Zemb[va], Vl[va])
            curves[frac]["raw"].append(sb.r2(Vl[te], f_raw(Zraw[te])))
            curves[frac]["emb"].append(sb.r2(Vl[te], f_emb(Zemb[te])))
        Vraw = np.roll(f_raw(Zraw), sb.LAG_BINS, 0)
        Vemb = np.roll(f_emb(Zemb), sb.LAG_BINS, 0)
        mv = te & moving
        keep = np.zeros(n, bool)
        keep[test_tr] = True
        tt = subset(trials, keep)
        h_true, secs = per_trial_hits(tt, tbin, P, V, cands)
        h_raw, _ = per_trial_hits(tt, tbin, P, Vraw, cands)
        h_emb, _ = per_trial_hits(tt, tbin, P, Vemb, cands)
        hits["true"].append(h_true)
        hits["raw"].append(h_raw)
        hits["emb"].append(h_emb)
        secs_all.append(secs)
        folds.append({
            "fold": k + 1, "test_trials": [int(test_tr[0]), int(test_tr[-1])],
            "pretrain_final_nll": hist[-1], "mean_rate_nll": base,
            "r2_all": {"raw": curves[1.0]["raw"][-1], "emb": curves[1.0]["emb"][-1]},
            "r2_movement": {"raw": sb.r2(V[mv], Vraw[mv]), "emb": sb.r2(V[mv], Vemb[mv])},
            "acc": {"true": h_true.mean(), "raw": h_raw.mean(), "emb": h_emb.mean()},
        })
        print(f"  R² all raw={folds[-1]['r2_all']['raw']:.3f} emb={folds[-1]['r2_all']['emb']:.3f} | "
              f"movement raw={folds[-1]['r2_movement']['raw']:.3f} emb={folds[-1]['r2_movement']['emb']:.3f} | "
              f"acc true={h_true.mean():.2f} raw={h_raw.mean():.2f} emb={h_emb.mean():.2f}", flush=True)

    secs = np.concatenate(secs_all)
    H = {k: np.concatenate(v) for k, v in hits.items()}
    rng = np.random.default_rng(0)
    T = secs.sum()

    def bps(h, s):
        return sb.webgrid_bps(n_targets, h.sum(), len(h) - h.sum(), s.sum())

    boot = {k: [] for k in H}
    diff = []
    for _ in range(2000):
        idx = rng.integers(0, len(secs), len(secs))
        for k in H:
            boot[k].append(bps(H[k][idx], secs[idx]))
        diff.append(H["emb"][idx].mean() - H["raw"][idx].mean())

    summary = {
        "dataset": "MC_Maze medium (DANDI 000139)", "folds": K, "trials_scored": int(len(secs)),
        "n_targets": int(n_targets), "selection_seconds": float(T),
        "runtime_s": round(time.time() - t0, 1),
        "r2_all_mean": {d: float(np.mean([f["r2_all"][d] for f in folds])) for d in ["raw", "emb"]},
        "r2_movement_mean": {d: float(np.mean([f["r2_movement"][d] for f in folds])) for d in ["raw", "emb"]},
        "label_efficiency_mean_r2": {str(fr): {d: float(np.mean(curves[fr][d])) for d in ["raw", "emb"]} for fr in FRACS},
        "selection": {
            k: {"accuracy": float(H[k].mean()), "bps": float(bps(H[k], secs)),
                "bps_95ci": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]}
            for k in H},
        "accuracy_diff_emb_minus_raw_95ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "per_fold": folds,
    }
    with open(os.path.join(out_dir, "stage_b_crossval.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps({k: summary[k] for k in ["r2_all_mean", "r2_movement_mean", "label_efficiency_mean_r2",
                                              "selection", "accuracy_diff_emb_minus_raw_95ci", "runtime_s"]}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
