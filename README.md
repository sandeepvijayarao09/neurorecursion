# NeuroRecursion

**Can a brain-computer interface decoder keep working from day to day without the user recalibrating it?**

A BCI decoder turns spikes from motor cortex into cursor or arm movement. It degrades every day: electrodes die, channels get noisy, and neural activity drifts. Today the usual fix is regular recalibration with labelled data from the user, about 55 minutes a week by Neuralink's public account (see [`notes/signal-path.md`](notes/signal-path.md)).

I wanted to see how far I could get without touching the decoder at all. My approach trains it once, freezes it, and lets a small team of agents adapt the *signal around it* each day, with no labels. This repo is the record of that experiment, including the parts that didn't work.

Research prototype, not a medical device.

## What I found

| Question | Answer so far |
|---|---|
| Can label-free agents beat a frozen decoder on real drift? | **Yes, mostly.** On FALCON H1 (7 held-out days of human intracortical data) they beat it on **6 of 7 days**: mean velocity R² **−0.016 vs −0.062**. |
| Do they replace recalibration? | **Not yet.** Few-shot recalibration with labels still leads (**+0.106**). |
| Does it hold on sessions I never tuned on? | **Partly.** With 14 sessions split into dev (1–7) and test (8–14), test R² is **−0.015 vs −0.100** for the frozen decoder and **+0.087** for recalibration. That closes about **77%** of the gap with zero labels. |
| What matters most? | **The verifier.** With it switched off, the same agents wreck the decoder (R² **−1.277**). |
| Does an LLM planner help? | **Not yet.** Four Nemotron planners all land within noise of a rule-based planner. The result comes from the agents, the verifier and the memory, not from the LLM. |
| Does self-supervised pretraining help when labels are scarce? | **For R², yes:** 0.27 vs 0.18 with about 11 s of labels on MC_Maze. **For target selection, no:** accuracy dropped from 53% to 41%. |

Absolute R² values are low for every method. They are for comparing methods with each other, not against the FALCON leaderboard.

## How I got here, one question at a time

1. **How much data does the implant have to throw away?** ([`demo/RESULTS.md`](demo/RESULTS.md), Stage A)
   I ran spike detection on Neuralink's public compression-challenge recordings (743 channels at 19,531 Hz). Spike events need **0.12 Mbps** for 1,024 channels, against **200 Mbps** raw, about 1,600× less. The best generic lossless compression managed only **1.9×**. So spike detection, not compression, is what makes the wireless link possible.

2. **Does pretraining on unlabelled spikes help when calibration data is scarce?** (Stage B)
   I built a small spike-token encoder: per-channel embeddings, 2 Mamba-style state-space blocks, and masked auto-Poisson pretraining (see half the channels, predict the rest). Under 5-fold blocked cross-validation it beat raw-spike ridge regression at every amount of labelled data:

   | Labelled calibration data | Raw-spike ridge | Pretrained encoder |
   |---|---|---|
   | 2% (~11 s) | 0.18 | **0.27** |
   | 10% | 0.27 | **0.38** |
   | 100% | 0.37 | **0.43** |

   The surprise was that target selection got *worse* (−12 points, 95% CI −19 to −4). My best explanation is that a small low-frequency bias in decoded velocity integrates into endpoint error over a curved one-second reach. R² doesn't penalise that bias; the endpoint does.

3. **Can agents adapt around a frozen decoder?** ([`agentic/README.md`](agentic/README.md))
   This is v0, on simulated drift: dead channels, noisy channels and per-channel gain changes. The plumbing worked, and on days 1–2 the agents matched or beat daily recalibration without labels. On day 4, the verifier accepted harmful changes, because its score came from only 7 trials. That failure shaped every later version.

4. **Does it hold on real drift?** ([`agentic/EVALS.md`](agentic/EVALS.md))
   I moved to FALCON H1, which has real cross-day human data. The decoder is trained on 6 held-in days, then frozen. The agents adapt on the first 30% of each new day and are scored on the remaining 70%.

5. **Does remembering each user help?** ([`agentic/MEMORY_ARCHITECTURE.md`](agentic/MEMORY_ARCHITECTURE.md), [`agentic/EVALS_14_SESSIONS.md`](agentic/EVALS_14_SESSIONS.md))
   I added a per-user SQLite memory: an agent schedule chosen by Thompson sampling, a warm start from the last verified state, and rollback. I made every design decision on sessions 1–7 only. The kept change came from a specific failure: in session 4, a warm start stacked a second gain change on top and drove R² to −0.444. Now a remembered state is trusted only if it wins on *both halves* of today's data. That cut the worst session to −0.292.

6. **Does an LLM planner add anything?** ([`agentic/LLM_IN_THE_LOOP.md`](agentic/LLM_IN_THE_LOOP.md), [`agentic/PLANNER_COMPARISON.md`](agentic/PLANNER_COMPARISON.md))
   I tried a Nemotron planner that orders the agents each day, plus a reviewer that rewrites the playbook between versions. A held-out gate keeps a rewrite only if it improves R². The gate worked: it blocked every edit that didn't help. But no planner beat rules, and one early +0.013 result didn't reproduce over 3 more runs. I've corrected it in the planner comparison.

## What didn't work, and what I think it means

- **The label-free verifier can be fooled.** On the Feb 6 session, an input-gain change looked right by the verifier's score but hurt decoding. Most planners hit the same failure day (worst-day mean −0.378), because they all see the same score.
- **Changes I selected on dev sessions made the system safer, not more accurate, on test sessions.** The v3 test score (−0.015) is slightly behind v1 (−0.011).
- **The LLM reviewer re-proposed edits that had already been reverted**, even when the prompt said not to. That needs a hard filter, not an instruction.
- **Delayed credit,** where today re-checks yesterday's changes, didn't help with about 15 s of adaptation data per session.
- **I found a bug after seeing test results:** an agent could be applied twice. I fixed it, and kept the buggy run in the repo (`agentic/out_falcon/falcon_h1_eval_smooth_run1_bug.json`, R² −0.069) so the before and after are both visible.

## Questions I'm still working on

- **A better gate.** Would the frozen encoder's own masked-Poisson prediction loss work as a label-free consistency check, so the verifier stops accepting gain changes that look right but hurt?
- **Closed loop.** Offline, the decoder can't correct its own drift. With a user in the loop, both sides adapt to each other. Does a frozen decoder with a translating layer stay stable under that co-adaptation?
- **A new user, out of the box.** If one frozen decoder can be kept usable across days by adapting the input, can the same translating layer map a *new* person's signals onto it?
- **Pseudo-labels.** ReFIT-style labels (the direction from cursor to target) would give the intent agent something to learn from in closed loop.

## How it works

```
 slow loop (per day): planner (rules or LLM) orders the agents, reads verifier scores and memory
                         │
 spikes ─► channel-health agent ─► mask dead/noisy channels, rescale the rest ─┐
        ├► alignment agent      ─► input gain match | latent CORAL transform   ┤ proposals
        └► intent agent         ─► pseudo-labels → rotation/gain on output     ▼
                         verifier: accept / reject (label-free, split-half check)
                                                 ▼
                 frozen decoder (never modified) ─► cursor velocity
                         ▲                                   │
                         └── per-user memory: what helped, last verified state, rollback
```

All per-user customisation lives in a small, versioned, reversible state: channel weights, a latent affine transform and a 2×2 output map. Agents never see hand kinematics; ground truth is only used for scoring.

## Repo map
| Path | What |
|---|---|
| `notes/` | My reading notes on how a Neuralink implant turns voltage into a cursor, every fact tagged with its public source |
| `demo/` | Stage A (signal → spikes → bandwidth) and Stage B (spikes → pretrained encoder → cursor), with `RESULTS.md` |
| `agentic/agents.py` | Frozen decoder wrapper, adaptation state, tool agents, verifier and orchestrator |
| `agentic/memory.py`, `brain.py`, `brain_v2.py` | Per-user memory and the memory-driven orchestrator, including the split-half safeguards |
| `agentic/recursive.py` | Version N+1 loop: planner, definition-of-done checks, reviewer and held-out gate |
| `agentic/llm.py`, `check_crusoe.py` | LLM client (Nemotron via Crusoe Managed Inference), with a rule-based fallback |
| `agentic/eval_*.py`, `compare_planners.py` | Evaluation harnesses, each with a write-up in `agentic/*.md` |
| `viz/` | Standalone pages: existing system vs agent system, evals and the 14-session chart. Open `viz/evals.html` in a browser |

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/get_data.sh
NO_LLM=1 .venv/bin/python agentic/eval_14_sessions.py data/falcon_h1 agentic/out_14sessions
MASTER=smooth .venv/bin/python agentic/eval_falcon_h1.py data/falcon_h1 agentic/out_falcon
```
The LLM planner is optional. Without an API key it falls back to rules, and the main results don't depend on it. To run it, put a Crusoe key in `~/.config/crusoe/api_key` and check the connection with `.venv/bin/python agentic/check_crusoe.py`.

## Data
All data is public and downloaded by `scripts/get_data.sh`; none is stored in this repo.
- **FALCON H1** (DANDI 000954): human, intracortical, multi-day
- **MC_Maze** (DANDI 000139): monkey motor cortex, reaching
- **Neuralink's 2024 compression-challenge recordings**, from a community mirror

## Origin
I started this on October 17, 2026, during the Crusoe × Cerebral Valley Recursive Agents Hackathon. That's why the optional LLM planner runs on NVIDIA Nemotron via Crusoe.
