"""Stage B — spikes -> pretrained encoder -> cursor velocity -> BPS.

A small-scale re-creation of the decoder recipe in Neuralink's Oct 2026 post
("Pretraining on 50,000 Hours of Unlabeled Brain Data"), run on the public
MC_Maze dataset (monkey motor cortex, DANDI 000139), because Neuralink has not
released any labelled data.

Recipe elements reproduced:
  * spikes as tokens with a learned per-channel embedding   (POYO-style)
  * a causal selective state-space backbone                  (Mamba-style)
  * self-supervised spatially masked auto-Poisson objective:
    see half the channels, predict the other half's spike counts
  * frozen encoder + small linear decoder head fit on few labels
Compared against the classic baseline: ridge regression on raw binned spikes.

Usage: python stage_b_spikes_to_cursor.py <mc_maze.nwb> <out_dir>
"""
import json
import os
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BIN_S = 0.02          # 20 ms bins
LAG_BINS = 4          # neural activity leads hand velocity by ~80 ms
HIST = 10             # ridge baseline uses 200 ms of spike history
WIN = 64              # encoder training window (1.28 s)
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def load_nwb(path):
    with h5py.File(path, "r") as f:
        st = f["units/spike_times"][:]
        idx = f["units/spike_times_index"][:]
        units = np.split(st, idx[:-1])
        beh = f["processing/behavior"]
        hv = beh["hand_vel/data"][:]
        ht = beh["hand_vel/timestamps"][:]
        hp = beh["cursor_pos/data"][:]   # cursor frame = target frame
        tr = f["intervals/trials"]
        trials = {k: tr[k][:] for k in tr.keys() if isinstance(tr[k], h5py.Dataset)
                  and tr[k].shape and tr[k].shape[0] == tr["start_time"].shape[0]}
        if "target_pos_index" in tr:
            tp, tpi = tr["target_pos"][:], tr["target_pos_index"][:]
            trials["target_pos_list"] = np.split(tp, tpi[:-1])
    return units, ht, hv, hp, trials


def bin_data(units, ht, hv, hp):
    t0, t1 = ht[0], ht[-1]
    edges = np.arange(t0, t1, BIN_S)
    X = np.stack([np.histogram(u, edges)[0] for u in units], 1).astype(np.float32)
    centers = edges[:-1] + BIN_S / 2
    ok = ~np.isnan(hv).any(1)
    V = np.stack([np.interp(centers, ht[ok], hv[ok, i]) for i in range(2)], 1).astype(np.float32)
    P = np.stack([np.interp(centers, ht[ok], hp[ok, i]) for i in range(2)], 1).astype(np.float32)
    return edges[:-1], X, V, P


# ----------------------------------------------------------------------------- model
class SelectiveSSM(nn.Module):
    """Minimal Mamba-style block: input-dependent (Δ, B, C), diagonal A, causal scan."""

    def __init__(self, d, n=16):
        super().__init__()
        self.inp = nn.Linear(d, 2 * d)
        self.conv = nn.Conv1d(d, d, 4, groups=d, padding=3)
        self.dt = nn.Linear(d, d)
        self.B = nn.Linear(d, n)
        self.C = nn.Linear(d, n)
        self.logA = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))
        self.D = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                       # x: [B, T, d]
        h_in = self.norm(x)
        u, z = self.inp(h_in).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))
        dt = F.softplus(self.dt(u))             # [B,T,d]
        Bm, Cm = self.B(u), self.C(u)           # [B,T,n]
        A = -torch.exp(self.logA)               # [d,n]
        h = torch.zeros(x.shape[0], x.shape[2], A.shape[1], device=x.device)
        ys = []
        for t in range(x.shape[1]):             # causal recurrence
            dA = torch.exp(dt[:, t, :, None] * A)
            h = dA * h + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.D * u[:, t])
        y = torch.stack(ys, 1) * F.silu(z)
        return x + self.out(y)


class SpikeEncoder(nn.Module):
    """Spikes-as-tokens: each spike adds its channel's learned embedding to the bin."""

    def __init__(self, n_ch, d=64, layers=2):
        super().__init__()
        self.ch_emb = nn.Parameter(torch.randn(n_ch, d) * 0.5)
        self.blocks = nn.ModuleList([SelectiveSSM(d) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Linear(d, n_ch)       # Poisson log-rate for every channel

    def embed(self, counts, mask=None):          # counts [B,T,C]
        if mask is not None:
            counts = counts * mask / 0.5   # keep input scale equal to unmasked inference
        x = counts @ self.ch_emb                 # sum of spike tokens per bin
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward(self, counts, mask):
        return self.readout(self.embed(counts, mask))


def pretrain(X_train, n_ch, epochs=30, batch=64):
    enc = SpikeEncoder(n_ch).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), 2e-3, weight_decay=1e-4)
    Xt = torch.tensor(X_train)
    starts = np.arange(0, len(Xt) - WIN, 4)   # heavily overlapping windows
    hist = []
    mean_rate = Xt.mean(0).clamp_min(1e-4)
    base = F.poisson_nll_loss(mean_rate.log().expand_as(Xt), Xt, log_input=True).item()
    print(f"  constant-rate baseline NLL {base:.4f}", flush=True)
    for ep in range(epochs):
        np.random.shuffle(starts)
        tot = 0.0
        for i in range(0, len(starts), batch):
            xb = torch.stack([Xt[s:s + WIN] for s in starts[i:i + batch]]).to(DEV)
            mask = (torch.rand(xb.shape[0], 1, n_ch, device=DEV) < 0.5).float()
            logr = enc(xb, mask)
            # spatially masked auto-Poisson: score only the hidden half
            nll = F.poisson_nll_loss(logr, xb, log_input=True, reduction="none")
            loss = (nll * (1 - mask)).sum() / (1 - mask).sum() / WIN
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        hist.append(tot / max(1, len(range(0, len(starts), batch))))
        print(f"  pretrain epoch {ep + 1:2d}  masked Poisson NLL {hist[-1]:.4f}", flush=True)
    return enc, hist, base


@torch.no_grad()
def embed_all(enc, X):
    enc.eval()
    Xt = torch.tensor(X)[None].to(DEV)
    # causal model: run in long chunks with warm-up context
    out, chunk, warm = [], 1024, 64
    for s in range(0, X.shape[0], chunk):
        a = max(0, s - warm)
        e = enc.embed(Xt[:, a:s + chunk])[0, s - a:]
        out.append(e.cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- decoders
def history(X, h=HIST):
    Z = [np.roll(X, k, 0) for k in range(h)]
    Z = np.concatenate(Z, 1)
    Z[:h] = 0
    return Z


def ridge_fit(Z, Y, lam):
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zs = (Z - mu) / sd
    W = np.linalg.solve(Zs.T @ Zs + lam * np.eye(Zs.shape[1]), Zs.T @ (Y - Y.mean(0)))
    return lambda Q: ((Q - mu) / sd) @ W + Y.mean(0)


def r2(y, yhat):
    return float(np.mean(1 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))


def best_ridge(Ztr, Ytr, Zva, Yva):
    best = None
    for lam in [1, 10, 100, 1e3, 1e4]:
        f = ridge_fit(Ztr, Ytr, lam)
        s = r2(Yva, f(Zva))
        if best is None or s > best[0]:
            best = (s, lam)
    return ridge_fit(np.concatenate([Ztr, Zva]), np.concatenate([Ytr, Yva]), best[1])


# ----------------------------------------------------------------------------- BPS
def webgrid_bps(n_targets, correct, incorrect, seconds):
    """Neuralink's formula with N_click = 1: log2(N-1) * max(c - i, 0) / time."""
    return np.log2(n_targets - 1) * max(correct - incorrect, 0) / seconds


def selection_eval(trials, tbin, P, Vdec, label, cands=None):
    """Offline target selection: integrate decoded velocity from movement onset
    to trial end; pick the candidate target whose direction best matches the
    decoded endpoint."""
    tp = trials["target_pos_list"]
    act = trials["active_target"].astype(int)
    true_t = np.array([tp[i][act[i]] for i in range(len(act))])
    if cands is None:
        cands = np.unique(np.round(true_t, -1), axis=0)
    correct = incorrect = 0
    secs = 0.0
    paths = []
    for i in range(len(act)):
        on, end = trials["move_onset_time"][i], trials["stop_time"][i]
        go = trials["go_cue_time"][i] if "go_cue_time" in trials else trials["start_time"][i]
        if np.isnan(on):
            continue
        a, b = np.searchsorted(tbin, on), min(np.searchsorted(tbin, end), len(tbin) - 1)
        if b - a < 5:
            continue
        p0 = P[a]
        path = p0 + np.cumsum(Vdec[a:b] * BIN_S, 0)
        # pick by direction of the decoded endpoint (robust to the magnitude
        # shrinkage that regression decoders show); same rule for every decoder
        d = path[-1] - p0
        cd = cands - p0
        cos = (cd @ d) / (np.linalg.norm(cd, axis=1) * np.linalg.norm(d) + 1e-9)
        guess = cands[np.argmax(cos)]
        truth = cands[np.argmin(((cands - true_t[i]) ** 2).sum(1))]
        ok = np.allclose(guess, truth)
        correct += ok
        incorrect += not ok
        secs += (on - go) + (tbin[b] - on)
        paths.append((P[a:b], path, true_t[i], ok))
    bps = webgrid_bps(len(cands), correct, incorrect, secs)
    return {"decoder": label, "n_targets": int(len(cands)), "trials": correct + incorrect,
            "correct": int(correct), "accuracy": correct / max(1, correct + incorrect),
            "seconds": secs, "bps": float(bps)}, paths, cands


# ----------------------------------------------------------------------------- main
def main(nwb, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    units, ht, hv, hp, trials = load_nwb(nwb)
    tbin, X, V, P = bin_data(units, ht, hv, hp)
    n_ch = X.shape[1]
    # neural activity at t predicts velocity at t + lag
    Vl = np.roll(V, -LAG_BINS, 0)
    print(f"units={n_ch} bins={len(X)} ({len(X) * BIN_S / 60:.1f} min) trials={len(trials['start_time'])}")

    # split by trials in time: first 70% train, next 10% val, last 20% test
    st = trials["start_time"]
    n_tr = len(st)
    cut = [st[int(n_tr * 0.7)], st[int(n_tr * 0.8)]]
    tr_m, va_m, te_m = tbin < cut[0], (tbin >= cut[0]) & (tbin < cut[1]), tbin >= cut[1]
    in_trial = np.zeros(len(tbin), bool)
    for s, e in zip(trials["start_time"], trials["stop_time"]):
        in_trial[(tbin >= s) & (tbin < e)] = True

    # --- self-supervised pretraining on ALL training-period spikes, no labels used
    print("Pretraining spike encoder (no behaviour labels)…")
    enc, hist, base_nll = pretrain(X[tr_m], n_ch)
    E = embed_all(enc, X)

    Zraw = history(X)
    Zemb = history(E)          # same 200 ms history window for a fair comparison
    results, curves = {}, {"label_fraction": [], "raw_r2": [], "emb_r2": []}
    tr_idx = np.flatnonzero(tr_m & in_trial)
    va = va_m & in_trial
    te = te_m & in_trial
    for frac in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
        sub = tr_idx[: max(50, int(len(tr_idx) * frac))]   # contiguous early slice = "calibration"
        f_raw = best_ridge(Zraw[sub], Vl[sub], Zraw[va], Vl[va])
        f_emb = best_ridge(Zemb[sub], Vl[sub], Zemb[va], Vl[va])
        curves["label_fraction"].append(frac)
        curves["raw_r2"].append(r2(Vl[te], f_raw(Zraw[te])))
        curves["emb_r2"].append(r2(Vl[te], f_emb(Zemb[te])))
        print(f"  labels {frac:5.0%} ({len(sub) * BIN_S:6.1f} s): raw-spike R²={curves['raw_r2'][-1]:.3f}  "
              f"embedding R²={curves['emb_r2'][-1]:.3f}")
        if frac == 1.0:
            dec_raw, dec_emb = f_raw(Zraw), f_emb(Zemb)

    # shift decoded velocity back onto the timeline of the hand
    Vraw = np.roll(dec_raw, LAG_BINS, 0)
    Vemb = np.roll(dec_emb, LAG_BINS, 0)

    moving = np.zeros(len(tbin), bool)
    for on, e in zip(trials["move_onset_time"], trials["stop_time"]):
        if not np.isnan(on):
            moving[(tbin >= on) & (tbin < e)] = True
    mv = te & moving
    r2_move = {"ridge_raw_spikes": r2(V[mv], Vraw[mv]), "pretrained_embeddings": r2(V[mv], Vemb[mv])}
    print("  movement-period R²:", r2_move, flush=True)
    np.savez_compressed(os.path.join(out_dir, "stage_b_decoded.npz"), t=tbin, V=V, Vraw=Vraw, Vemb=Vemb, P=P, test=te, moving=moving)

    te_trials = {k: (v[st >= cut[1]] if isinstance(v, np.ndarray) else
                     [x for x, s in zip(v, st) if s >= cut[1]]) for k, v in trials.items()}
    sel_true, paths_true, cands = selection_eval(te_trials, tbin, P, V, "true hand (upper bound)")
    sel_raw, paths_raw, _ = selection_eval(te_trials, tbin, P, Vraw, "ridge on raw spikes")
    sel_emb, paths_emb, _ = selection_eval(te_trials, tbin, P, Vemb, "pretrained encoder + linear head")

    results = {
        "dataset": "MC_Maze medium (DANDI 000139), Jenkins, motor/premotor cortex",
        "units": n_ch, "bin_ms": BIN_S * 1000, "lag_ms": LAG_BINS * BIN_S * 1000,
        "device": DEV, "runtime_s": round(time.time() - t_start, 1),
        "pretrain_loss": hist, "constant_rate_baseline_nll": base_nll,
        "velocity_r2_test": {"ridge_raw_spikes": curves["raw_r2"][-1], "pretrained_embeddings": curves["emb_r2"][-1]},
        "velocity_r2_test_movement_only": r2_move,
        "label_efficiency": curves,
        "offline_selection_bps": [sel_true, sel_raw, sel_emb],
        "bps_note": "Offline proxy: Neuralink's Webgrid formula (N_click=1) applied to reach-target "
                    "selection from integrated decoded velocity. Not closed-loop; not comparable to "
                    "Neuralink's published 11.32 BPS.",
    }
    with open(os.path.join(out_dir, "stage_b_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # --- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist, "o-", color="#8250df", label="encoder")
    ax[0].axhline(base_nll, color="#999", ls="--", label="mean-rate baseline")
    ax[0].legend()
    ax[0].set_title("Self-supervised pretraining\n(masked auto-Poisson, no labels)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("held-out-channel Poisson NLL")
    secs = np.array(curves["label_fraction"]) * len(tr_idx) * BIN_S
    ax[1].plot(secs, curves["raw_r2"], "o-", label="ridge on raw spikes", color="#6e7781")
    ax[1].plot(secs, curves["emb_r2"], "o-", label="pretrained embeddings", color="#1a7f37")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("seconds of labelled calibration data")
    ax[1].set_ylabel("test velocity R²")
    ax[1].set_title("Calibration efficiency")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "B1_pretraining_calibration.png"), dpi=130)
    plt.close(fig)

    w = te & (np.cumsum(te) < 400)
    tt = tbin[w] - tbin[w][0]
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for i, nm in enumerate(["x", "y"]):
        ax[i].plot(tt, V[w, i], color="k", lw=1.2, label="true hand")
        ax[i].plot(tt, Vraw[w, i], color="#6e7781", lw=0.9, label="ridge raw")
        ax[i].plot(tt, Vemb[w, i], color="#1a7f37", lw=0.9, label="pretrained")
        ax[i].set_ylabel(f"v{nm} (mm/s)")
    ax[0].legend(loc="upper right", ncol=3)
    ax[1].set_xlabel("s (held-out test trials)")
    ax[0].set_title("Decoded cursor velocity vs true hand velocity")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "B2_velocity.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, paths, ttl in [(ax[0], paths_raw, sel_raw), (ax[1], paths_emb, sel_emb)]:
        a.scatter(cands[:, 0], cands[:, 1], s=120, facecolors="none", edgecolors="#999")
        for real, dec, tgt, ok in paths[:40]:
            a.plot(real[:, 0], real[:, 1], color="k", lw=0.6, alpha=0.4)
            a.plot(dec[:, 0], dec[:, 1], color="#1a7f37" if ok else "#d1242f", lw=1)
        a.set_title(f"{ttl['decoder']}\n{ttl['accuracy']:.0%} correct · {ttl['bps']:.2f} BPS (offline)")
        a.set_aspect("equal")
    fig.suptitle("Cursor output: decoded reach paths (green = correct target, red = wrong; black = real hand)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "B3_cursor_selection.png"), dpi=130)
    plt.close(fig)

    print(json.dumps({k: results[k] for k in ["velocity_r2_test", "offline_selection_bps", "runtime_s"]}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
