"""Stage A — Neuralink raw signal -> spikes -> firing rates -> compression.

Runs on Neuralink's 2024 compression-challenge recordings (monkey motor cortex,
19,531 Hz, ~10-bit). Mirrors the on-implant steps described in Neuralink's
2019 paper and spike-detection patent: band-pass filter, threshold detection,
event packetisation, and the bandwidth those choices buy.

Usage: python stage_a_signal_to_spikes.py <wav_dir> <out_dir>
"""
import glob
import json
import lzma
import os
import sys
import wave
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

FS = 19531
THRESH_K = 4.5          # threshold = -k * robust noise sigma (standard in iBCI work)
REFRACTORY_MS = 1.0
SNIPPET = (-10, 22)     # samples around the trough (~0.5 ms pre, ~1.1 ms post)
BIN_MS = 20             # firing-rate bin, as used for decoders
N1_CHANNELS, N1_RADIO_BPS = 1024, 1e6


def load(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    return x


def detect(x, sos_causal, sos_offline):
    # Causal filter is what an implant can run; zero-phase used for clean snippets.
    xf = sosfilt(sos_causal, x)
    sigma = np.median(np.abs(xf)) / 0.6745
    thr = -THRESH_K * sigma
    below = np.flatnonzero((xf[1:] < thr) & (xf[:-1] >= thr)) + 1
    refr = int(REFRACTORY_MS * FS / 1000)
    spikes, last = [], -refr
    for i in below:
        if i - last >= refr:
            seg = xf[i:i + refr]
            spikes.append(i + int(np.argmin(seg)))
            last = i
    return xf, np.array(spikes, dtype=np.int64), sigma, thr


def ten_bit(x):
    # Data are stored in 16-bit but quantised in ~64-count steps (10-bit ADC).
    return np.round(x / 64).astype(np.int16)


def main(wav_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    sos_c = butter(2, [300, 6000], btype="band", fs=FS, output="sos")
    sos_o = butter(4, [300, 6000], btype="band", fs=FS, output="sos")

    per_ch, all_snips = [], []
    raw_bytes = zlib_bytes = lzma_bytes = 0
    total_samples = total_spikes = 0
    example = None

    for k, f in enumerate(files):
        x = load(f)
        xf, sp, sigma, thr = detect(x, sos_c, sos_o)
        dur = len(x) / FS
        rate = len(sp) / dur
        per_ch.append({"file": os.path.basename(f), "dur_s": dur, "n_spikes": int(len(sp)),
                       "rate_hz": rate, "noise_sigma": float(sigma)})
        total_samples += len(x)
        total_spikes += len(sp)

        # lossless baselines on the 10-bit samples (delta + generic coders)
        q = ten_bit(x)
        d = np.diff(q, prepend=q[:1]).astype(np.int16).tobytes()
        raw_bytes += len(x) * 10 / 8
        zlib_bytes += len(zlib.compress(d, 9))
        if k < 150:  # lzma is slow; estimate ratio on a subset
            lzma_bytes += len(lzma.compress(d, preset=9))

        xo = sosfiltfilt(sos_o, x)
        for s in sp:
            a, b = s + SNIPPET[0], s + SNIPPET[1]
            if a >= 0 and b <= len(xo):
                all_snips.append(xo[a:b])
        if example is None and 20 < rate < 80:
            example = (os.path.basename(f), x, xf, sp, thr)

    rates = np.array([c["rate_hz"] for c in per_ch])
    secs = total_samples / FS

    # --- bandwidth: what the N1 radio would need under each representation ---
    raw_bps_ch = FS * 10
    ts_bits = int(np.ceil(np.log2(FS * BIN_MS / 1000)))      # timestamp within a 20 ms frame
    ch_bits = int(np.ceil(np.log2(N1_CHANNELS)))
    spike_event_bps_ch = rates.mean() * (ts_bits + ch_bits)  # spike time + channel id
    bins_per_s = 1000 / BIN_MS
    max_count = max(1, int(np.ceil(rates.max() * BIN_MS / 1000)))
    count_bits = int(np.ceil(np.log2(max_count + 1)))
    binned_bps_ch = bins_per_s * count_bits                  # per-channel spike count per bin
    lossless_ratio_zlib = raw_bytes / zlib_bytes
    sub_raw = sum(len(load(f)) for f in files[:150]) * 10 / 8
    lossless_ratio_lzma = sub_raw / lzma_bytes

    bw = {
        "raw_10bit": raw_bps_ch,
        "lossless_lzma": raw_bps_ch / lossless_ratio_lzma,
        "binned_counts_20ms": binned_bps_ch,
        "spike_events": spike_event_bps_ch,
    }
    n1 = {k: v * N1_CHANNELS for k, v in bw.items()}

    summary = {
        "files": len(files), "recording_minutes": secs / 60, "fs_hz": FS,
        "filter": "Butterworth band-pass 300-6000 Hz (causal 2nd-order for detection)",
        "threshold": f"-{THRESH_K} x MAD noise sigma, {REFRACTORY_MS} ms refractory",
        "total_spikes": int(total_spikes),
        "rate_hz": {"mean": float(rates.mean()), "median": float(np.median(rates)),
                    "p10": float(np.percentile(rates, 10)), "p90": float(np.percentile(rates, 90))},
        "lossless_ratio": {"delta+zlib": lossless_ratio_zlib, "delta+lzma(subset)": lossless_ratio_lzma},
        "bits_per_s_per_channel": bw,
        "N1_1024ch_Mbps": {k: v / 1e6 for k, v in n1.items()},
        "compression_vs_raw": {k: raw_bps_ch / v for k, v in bw.items()},
        "fits_1Mbps_radio": {k: bool(v <= N1_RADIO_BPS) for k, v in n1.items()},
    }
    with open(os.path.join(out_dir, "stage_a_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "stage_a_per_channel.json"), "w") as fh:
        json.dump(per_ch, fh)

    # --- figures ---
    name, x, xf, sp, thr = example
    t = np.arange(len(x)) / FS * 1000
    w = slice(0, int(0.5 * FS))
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t[w], x[w], lw=0.5, color="#555")
    ax[0].set_title(f"1. Raw broadband (Neuralink data, {name[:8]}…, 19.5 kHz, 10-bit)")
    ax[1].plot(t[w], xf[w], lw=0.5, color="#1f6feb")
    ax[1].axhline(thr, color="#d1242f", ls="--", lw=1, label=f"threshold −{THRESH_K}σ")
    sw = sp[sp < w.stop]
    ax[1].plot(t[sw], xf[sw], "v", color="#d1242f", ms=5)
    ax[1].legend(loc="upper right")
    ax[1].set_title("2. Band-pass 300–6000 Hz + threshold crossings")
    ax[2].eventplot(t[sw], colors="k", lineoffsets=0.5, linelengths=0.8)
    ax[2].set_yticks([])
    ax[2].set_title("3. Spike events → what is radioed off the implant")
    ax[2].set_xlabel("time (ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "A1_signal_to_spikes.png"), dpi=130)
    plt.close(fig)

    snips = np.array(all_snips[:20000])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ts = (np.arange(SNIPPET[0], SNIPPET[1]) / FS) * 1000
    for s in snips[np.random.default_rng(0).choice(len(snips), min(300, len(snips)), replace=False)]:
        ax[0].plot(ts, s, color="#1f6feb", alpha=0.05)
    ax[0].plot(ts, np.median(snips, 0), color="k", lw=2)
    ax[0].set_title(f"Detected waveforms (n={len(all_snips):,}, 300 shown)")
    ax[0].set_xlabel("ms from trough")
    ax[1].hist(rates, bins=40, color="#8250df")
    ax[1].set_title("Threshold-crossing rate per channel")
    ax[1].set_xlabel("Hz")
    ax[1].set_ylabel("channels")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "A2_waveforms_rates.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["Raw 10-bit", "Best lossless\n(delta+LZMA)", "20 ms spike\ncounts", "Spike events\n(time+chan)"]
    vals = [n1[k] / 1e6 for k in bw]
    bars = ax.bar(labels, vals, color=["#6e7781", "#9a6700", "#1f6feb", "#1a7f37"])
    ax.axhline(1, color="#d1242f", ls="--", label="N1 radio ≈ 1 Mbps")
    ax.set_yscale("log")
    ax.set_ylabel("Mbps for 1,024 channels (log)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}", ha="center")
    ax.legend()
    ax.set_title("Why the implant must send spikes, not raw signal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "A3_bandwidth.png"), dpi=130)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
