# v0: the first version of the adaptation layer

Goal: train the decoder **once**, then let agents adapt *around* the frozen model each day, so users don't recalibrate and the model isn't retrained.

```
 slow loop (per day): Orchestrator: plans agents, reads verifier scores, audit log
                         │
 spikes ─► ChannelHealthAgent ─► mask dead/noisy channels, rescale the rest ─┐
        ├► AlignmentAgent     ─► input gain match  |  latent CORAL transform  ┤ proposals
        └► IntentAgent        ─► pseudo-labels (direction to known target)   ┤
                                  → rotation/gain on decoded velocity        ▼
                        Verifier ─ accept / reject / iterate (label-free)
                                                 ▼
                 FrozenMaster (day-0 encoder + head, never modified) ─► cursor
                        ▲                                   │
                        └── recurse with accepted state, ≤ N=3 loops
```

| File | Contents |
|---|---|
| `agents.py` | `AdaptState`, `FrozenMaster`, the 3 agents, `Verifier`, `Orchestrator` |
| `run_agentic.py` | Day-0 training, 4 simulated drift days, 3-way comparison |
| `out/audit_log.jsonl` | Every proposal with before/after scores, decision and reason |
| `out/agentic_results.json`, `out/agentic_days.png` | Per-day results |

Run:
```bash
.venv/bin/python agentic/run_agentic.py data/mc_maze/mc_maze_medium_train.nwb agentic/out
```
It takes about 1 min.

## Design rules
- **The master model is frozen.** All customisation lives in a small `AdaptState`: channel weights, a latent affine transform, and a 2×2 output map. That state is versioned, logged, and reversible.
- **Agents never see hand kinematics.** They only use spikes, task timing (go cue) and the known target position. Ground truth appears only in the evaluation block.
- **The verifier is label-free.** Its main score is the cosine between decoded velocity and the direction to the target in the movement window, measured on a verify slice the agents didn't fit on. A speed-ratio guard rejects decoder collapse.
- **State carries over between days.** The orchestrator stops when a loop accepts nothing, or after 3 loops.

## First run (MC_Maze, drift that grows each day: dead channels, noisy channels, per-channel gain changes)
| Day | Frozen R² | Agentic R² | Daily recalibration R² | Accepted by verifier |
|---|---|---|---|---|
| 1 | 0.06 | **0.09** | 0.06 | latent CORAL, channel mask |
| 2 | 0.05 | 0.06 | 0.06 | channel mask |
| 3 | 0.02 | 0.00 | 0.05 | nothing (all rejected) |
| 4 | −0.00 | −0.09 | −0.01 | input gain, channel mask (67 masked) |

**Honest read:**
- The plumbing works: agents propose, the verifier decides, recursion runs, and everything is logged (29 audit records).
- Days 1–2 beat the frozen model and match daily recalibration without using any labels.
- On day 4 the verifier accepted harmful changes. Its proxy (intent cosine on only 7 trials) is too noisy, and the masking threshold flagged channels that weren't actually faulty.
- Absolute R² is low for every condition: a model trained on early trials only reaches about 0.16 on later ones even without simulated drift. That natural nonstationarity is exactly the problem this layer exists to solve.

## Next steps (v1)
1. **Stronger verifier:**
   - add the frozen encoder's masked-Poisson NLL as an input-consistency score
   - use bootstrap confidence intervals over trials instead of a fixed margin
   - roll back the previous day's state if it scores worse today
2. **Channel health:** decide using rate *and* waveform/Fano statistics, and cap how many channels can be masked per day.
3. **Intent agent:** use ReFIT-style pseudo-labels (direction from cursor to target at every bin) in closed loop. Debug the large rejections in the Procrustes step.
4. **LLM orchestrator (optional):** swap the rule-based `plan()` for a Claude call that reads the audit log and verifier scores and picks the next agent. Agents and verifier stay deterministic tools.
5. **Real multi-day data:** for example the Neural Latents Benchmark DMFC/MC_RTT sessions or the BrainGate multi-day sets, instead of simulated drift.
