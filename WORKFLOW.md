# Neuralink: brain signal → output workflow

Compiled 2026-10-17 from public sources (see `pipeline_data.json` for every fact tagged with its source and confidence).

```
 Motor cortex        N1 implant (in skull)                        External device (Neuralink App)          Output
 ───────────    ───────────────────────────────────────────    ─────────────────────────────────────    ─────────────
 Intent to   →  Threads  →  Amplify/filter  →  Spike/feature  →  Radio  →  Pretrained   →  Decoder  →  Post-proc  →  Cursor, clicks
 move hand      1,024 el.   10-bit ADC          detection +       ~1 Mbps   encoder         heads       / UI modes     typing (40 WPM)
 (spikes)       64 threads  ~20 kHz/ch          compression                 (Mamba, SSL)    (vel/click)               robot arm, speech*
                            ≈200 Mbps raw       (>200× needed)
```
\*speech (VOICE) and vision (Blindsight) are in trials / development.

## Stage by stage

**1. Signal source.** Electrodes go into the hand-knob area of motor cortex (precentral gyrus), located before surgery with fMRI while the patient attempts hand movement. Neurons there fire when a movement is intended, even with a severed spinal cord. One implant on one side of the brain still yields decodable signals for both hands.

**2. Sensing (threads + R1 robot).** The N1 has 1,024 electrodes on 64 polymer threads thinner than a hair. A robot inserts them because they're too fine to place by hand. Next generation: about 3,000 electrodes, plus changes to help threads stay in place.

**3. Analog front end + ADC.** Custom chips amplify and filter each channel and digitise it. In the 2019 paper's chip: 42.9–59.4 dB gain, 3 Hz–27 kHz bandwidth, 5.9 µVrms noise, 10-bit ADC at 19.3 kHz, 5.2 µW per channel. For the N1 that comes to 1024 × 20 kHz × 10 bit ≈ **205 Mbps** of raw data.

**4. On-implant reduction (the bottleneck).** The radio can only send about **1 Mbps**, so the data has to shrink by more than 200×. It also has to happen in under 1 ms on less than 10 mW. Lossless compression of the raw signal tops out around 4–5×, so the implant sends **spike events / neural features** instead of waveforms. Neuralink's patent describes how: filter the signal, fit a model, compare against thresholds, and package detected spikes with timestamps. After threads pulled out of the first patient's brain, Neuralink retuned this step to pick up broader "population" activity.

**5. Wireless + power.** The data goes over a low-power radio (usually described as BLE-class) to a phone or laptop. The battery charges inductively through the scalp, and there are no connectors. The 2019 research rig was wired (USB-C to a 10 GbE base station).

**6. Decoding (ML, off-implant).**
- *Old approach:* a supervised decoder per user and per feature, trained on calibration tasks. It degraded because neural signals drift, so users spent about 10 min/day (about 55 min/week) recalibrating.
- *Oct 2026 approach:* a self-supervised **neural encoder pretrained on >50,000 h of unlabeled data**:
  - each spike is a token with a per-channel embedding (POYO-style), fed through a Perceiver input layer into a **Mamba** state-space backbone, which is causal and low-latency;
  - the training objective is *spatially masked auto-Poisson regression*: see half the channels, predict the other half;
  - small decoder heads on the stable embeddings output cursor velocity, click probabilities, finger/key intent, or 7+ robot-arm degrees of freedom.
- *Results:* 30 s of labeled data now does what 3.5 min of raw-spike calibration used to. Decoders last more than a week, and one still worked 20 months later. A model trained on one participant transferred to another with 99% of its weights frozen.

**7. Post-processing / UI.** Smoothing, low-velocity suppression and click filtering were used before. The pretrained decoders made several of these unnecessary, which removed lag. A mode switcher maps intents to left/right/middle click, scroll, zoom, pan and drag.

**8. Outputs.** Cursor and multi-click control of computers and phones (games, CAD, Illustrator). Ten-finger "mind keyboard" typing at up to 40 WPM. Robotic-arm control (CONVOY). Next up: speech at about 140 WPM (VOICE) and vision (Blindsight).

## Performance (bits per second on Webgrid)
`BPS = log2(N_click · grid² − 1) · max(correct − incorrect, 0) / 60`

| When | BPS | Note |
|---|---|---|
| Early 2024 | 4.6 → 8.0 | P1 first session → May 2024 |
| — | 8–10 | Able-bodied person using a mouse |
| Before Oct 2026 | 10.39 | Previous record |
| Oct 2026 | **11.32** | P15, pretrained decoder |

Scale: 21 participants as of Jan 2026, with zero serious device-related adverse events reported.

## Latest status (checked 2026-10-17)

**Latest model: Oct 1 2026 pretrained encoder.** This is Neuralink's newest published decoder.
- Each participant gets their own self-supervised encoder: spike tokens feed a Perceiver input layer, then a Mamba backbone.
- It is pretrained by masked prediction across channels, using >50,000 h of unlabeled data.
- Small decoder heads sit on top for cursor, click and robot-arm control.
- Results: an 11.32 BPS record, calibration cut to as little as 10 min/week, and decoders that last weeks.
- Limits: all live results use models pretrained on one participant's data. Pooled multi-participant models aren't better online yet, and the next stated goals are one-shot or zero-shot calibration.
- **No weights, code or paper have been released.**

**Latest data.**
- About 50k+ h of recorded neural data, held internally.
- **No public Neuralink neural dataset exists.** The only open-source release is the `datarepo` catalog library (June 2025), which Neuralink says is "not a release of any Neuralink data". The compression-challenge sample data is no longer online.
- The Hugging Face "neuralink" account appears unrelated to the company.
- Participants: 21 (Neuralink, Jan 2026); about 27 per an independent tracker (Oct 2026).

**Latest outputs and hardware.**
- **Speech (VOICE, NCT07224256, UT Southwestern):** first thought-to-speech demo in Mar 2026. In a Sep 16 2026 video a participant produced "I love you" as speech in their pre-illness voice. It works by decoding phonemes and feeding them to a personalised text-to-speech model, with a delay of several seconds. No accuracy or latency figures are published (secondary reports).
- **Surgery:** first through-the-dura insertion, May 2026, Toronto (secondary report).
- **Not yet confirmed:** a 3,000-electrode implant in a human, or a first human Blindsight implant.

Updated workflow: steps 1–5 are unchanged. Step 6 is now *pretrained encoder → task heads*. A new output branch, *speech-cortex spikes → phoneme decoder → personal TTS voice*, is in trials.

## Confidence / gaps
- Stages 1, 2, 5, 6, 7, 8 come from Neuralink's own posts or a peer-reviewed paper. The exact on-implant algorithm in today's N1 (stage 4) isn't public. It's inferred from a patent, the compression-challenge numbers (that page is now offline), and community analysis.
- The radio type and the N1 ASIC specs aren't officially published. The chip numbers above are from the 2019 paper's research chip.
- All performance figures are reported by the company and haven't been independently checked.

## Sources
- Musk E, Neuralink. *An Integrated Brain-Machine Interface Platform With Thousands of Channels.* J Med Internet Res 2019;21(10):e16194. doi:10.2196/16194 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6914248
- Neuralink, PRIME Study Progress Update (2024-04-12) — https://neuralink.com/updates/prime-study-progress-update/
- Neuralink, PRIME Update — User Experience (2024-05-08) — https://neuralink.com/updates/prime-study-progress-update-user-experience/
- Neuralink, PRIME Update — Second Participant (2024-08-21) — https://neuralink.com/updates/prime-study-progress-update-second-participant/
- Neuralink, A Year of Telepathy (2025-02-05) — https://neuralink.com/updates/a-year-of-telepathy/
- Neuralink, Two Years of Telepathy (2026-01-28) — https://neuralink.com/updates/two-years-of-telepathy/
- Neuralink, Pretraining on 50,000 Hours of Unlabeled Brain Data (2026-10-01) — https://neuralink.com/updates/pretraining-on-50000-hours/
- Neuralink, Technology — https://neuralink.com/technology/
- Neuralink Compression Challenge (offline) — https://content.neuralink.com/compression-challenge/README.html
- US 2021/0012909 A1, Real-time neural spike detection — https://patents.justia.com/patent/20210012909
- Spikey (community compression analysis) — https://git.dominik-roth.eu/dodox/Spikey
- Neuralink, Speech Restoration / VOICE — https://neuralink.com/trials/speech-restoration/
- Neuralink, datarepo (2025-06-22) — https://neuralink.com/updates/datarepo/
- Yahoo Tech, imagined speech → audible words (2026-09-18) — https://tech.yahoo.com/ai/meta-ai/articles/neuralink-brain-implant-turned-imagined-160150476.html
- Benzinga, "Neuralink is restoring speech" (2026-03) — https://www.benzinga.com/markets/tech/26/03/51449305/elon-musk-says-neuralink-is-restoring-speech-as-brain-chip-converts-silent-brain-signals-into-audible-words
- Neura Pod, Neuralink Update — 2026 (independent) — https://www.neurapod.com/blog/neuralink-update-2026
- Storyboard18, Blindsight first human implant prep — https://storyboard18.com/digital/neuralink-readies-first-human-implant-of-vision-restoring-blindsight-device-88432.htm
