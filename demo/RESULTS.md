# Demo: brain signal → spikes → cursor (Neuralink pipeline)

Run on 2026-10-17. Commands are listed at the bottom.

## Stage A: Neuralink data, signal → spikes → compression
Data: Neuralink compression-challenge recordings: 743 channels × about 5 s, 19,531 Hz, 10-bit, monkey motor cortex, 62.6 min in total.

Steps: 300–6000 Hz band-pass → threshold at −4.5σ (MAD estimate) with a 1 ms refractory period → spike events.

| Representation | Mbps for 1,024 ch | vs raw | Fits ~1 Mbps radio? |
|---|---|---|---|
| Raw 10-bit | 200.0 | 1× | no |
| Best generic lossless (delta + LZMA) | 106.2 | 1.9× | no |
| 20 ms spike counts | 0.15 | ~1,300× | **yes** |
| Spike events (time + channel) | 0.12 | ~1,600× | **yes** |

- 23,841 spikes detected. Mean rate is 6.3 Hz per channel and the median is 0.6 Hz, so many channels are quiet.
- A few channels above 200 Hz are probably noise or artifacts.
- **Takeaway:** lossless compression gets nowhere near the ~200× the radio needs. Spike detection is the step that makes wireless transmission possible.

Figures: `out/A1_signal_to_spikes.png`, `out/A2_waveforms_rates.png`, `out/A3_bandwidth.png`.

## Stage B: public MC_Maze data, spikes → pretrained encoder → cursor
Data: MC_Maze medium (DANDI 000139): 152 units, 12.9 min, 250 reaching trials. Neuralink has released no labelled data, so this stage uses a public dataset.

Model (Neuralink's Oct 2026 recipe at small scale):
- spikes as tokens with per-channel embeddings
- 2 causal Mamba-style selective state-space blocks
- **self-supervised masked auto-Poisson pretraining** (see 50% of channels, predict the rest; no labels)
- frozen encoder → linear velocity head

Baseline: ridge regression on raw binned spikes. Both decoders get the same 200 ms of history.

Evaluation: 5-fold blocked cross-validation, with the encoder re-pretrained in each fold on training spikes only.

**Velocity decoding (test R², mean of 5 folds)**
| Labelled calibration data | Raw-spike ridge | Pretrained encoder |
|---|---|---|
| 2% (~11 s) | 0.18 | **0.27** |
| 10% (~54 s) | 0.27 | **0.38** |
| 100% (~9 min) | 0.37 | **0.43** |
| 100%, movement periods only | 0.47 | **0.51** |

Pretraining improved decoding in 4 of 5 folds. The gain is largest with little calibration data, which matches the effect Neuralink reports, at a much smaller scale.

**Cursor → target selection → BPS (offline proxy, 250 trials, 8 targets)**
| Decoder | Accuracy | BPS (95% CI) |
|---|---|---|
| True hand (upper bound) | 87% | 1.33 (1.18–1.47) |
| Raw-spike ridge | 53% | 0.11 (0–0.33) |
| Pretrained encoder | 41% | 0.00 (0–0) |

**Negative result:** the pretrained decoder has better velocity R² but *worse* target selection: −12 points, 95% CI −19 to −4. The likely cause is that integrating velocity over a ~1 s curved maze reach builds up the decoder's low-frequency drift. R² doesn't penalise that drift, but the endpoint does. Neuralink's closed-loop users correct drift in real time, and this offline test can't.

These BPS numbers are **not comparable** to Neuralink's 11.32 BPS:
- this is open-loop, not closed-loop
- it uses 152 units, not 1,024 channels
- it uses 13 min of pretraining data, not 50,000 h
- it is a reach task, not Webgrid

Figures: `out/B1_pretraining_calibration.png`, `out/B2_velocity.png`, `out/B3_cursor_selection.png`. Raw numbers: `out/stage_a_summary.json`, `out/stage_b_results.json` (single split), `out/stage_b_crossval.json`.

## Reproduce
```bash
.venv/bin/python -I demo/stage_a_signal_to_spikes.py data/raw/data demo/out
.venv/bin/python demo/stage_b_spikes_to_cursor.py data/mc_maze/mc_maze_medium_train.nwb demo/out
.venv/bin/python demo/stage_b_crossval.py data/mc_maze/mc_maze_medium_train.nwb demo/out
```
Runtimes on Apple GPU (MPS): Stage A ~1 min, single split ~95 s, cross-validation ~8 min.
