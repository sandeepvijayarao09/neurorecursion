# Memory layer and agent architecture

Goal: the agent team gets better for each user the longer it runs, and every decision comes from that user's recorded evidence.

```
             ┌──────────────────── UserMemory (SQLite, per user) ────────────────────┐
             │ episodic   episodes, proposals        what happened, every verdict      │
             │ semantic   channel_stats (EWMA,flags)  which channels are unreliable     │
             │            agent_beliefs (Beta α,β)    how often each agent helps        │
             │ procedural states                      last verified adaptation state    │
             │ notes      one note per topic           lean context for the LLM planner │
             └──────▲──────────────┬─────────────────────┬───────────────────▲─────────┘
                    │ write        │ read                │ read              │ consolidate
 spikes ─► 1 SENSE ─┘    2 RECALL ─┘ warm start vs fresh  3 SCHEDULE         │
                         (verifier picks; = rollback)     Thompson sampling, │
                                                          prune weak agents  │
                         4 ACT + VERIFY: each agent proposes once ───────────┘ 5 CONSOLIDATE
                         label-free verifier accepts only improvements          beliefs, state, notes
                                         │
                                         ▼
                          adaptation state ─► FROZEN decoder ─► outputs
```

| File | Role |
|---|---|
| `memory.py` | `UserMemory`: schema, read/write API, `thompson_order()`, `consolidate()`, and `summary()` for the LLM |
| `brain.py` | `MemoryOrchestrator.run_day()`, the 5 steps above; `ChannelMemoryAgent` (masks chronic channels, restores recovered ones) |
| `eval_memory_h1.py` | Chronological evaluation on real FALCON H1 days |
| `out_memory/user_memory.sqlite` | The memory after 7 days (open with any SQLite viewer) |

## How decisions are made from data
- **Which agents run, and in what order:** Thompson sampling on each agent's track record for this user (a Beta posterior updated from verifier verdicts). Agents whose success rate stays below 20% after 3 or more tries are pruned.
- **Where the day starts:** both the remembered state and a fresh one are scored on today's unlabelled verify slice, and the better one wins. If yesterday's state no longer fits, this acts as an automatic rollback.
- **Which channels to mask:** today's flags combined with long-run reliability (`1 − flags / (seen + 1)`). Channels that recover are restored.
- **Optional LLM planner:** Nemotron reorders the agents using `memory.summary()`. Without `CRUSOE_API_KEY`, the data-driven schedule above is used as is.

## Results: FALCON H1, 7 held-out days processed in date order
| Day | Frozen | Stateless agents | **Memory agents** | What memory did |
|---|---|---|---|---|
| Jan 26 | 0.035 | 0.198 | 0.157 | first day, nothing remembered yet |
| Jan 27 | −0.203 | −0.067 | −0.067 | |
| Jan 29 | −0.031 | 0.055 | 0.055 | |
| Feb 2 | −0.138 | −0.031 | **0.005** | warm-started from Jan 29 state |
| Feb 3 | 0.019 | 0.114 | 0.114 | pruned latent alignment (17% success) |
| Feb 6 | −0.090 | −0.378 | −0.378 | same failure as stateless |
| Feb 9 | −0.027 | 0.001 | **0.014** | |
| **Mean** | −0.062 | −0.016 | **−0.014** | memory − stateless: +0.001 (CI −0.016, +0.016) |

What memory learned about this user (`out_memory/memory_eval.json`):
- **Agent track record:** input gain helps 67% of the time, channel memory 44%, latent alignment 17% (now pruned).
- **Bad channels:** 6 channels are chronically unreliable: 14, 24, 90, 114, 149, 152.

**Honest read:**
- **The machinery works:** it builds a user profile, prunes the agent that doesn't help, warm-starts, and keeps memory lean.
- **No measurable accuracy gain yet:** the difference is +0.001 R². Seven days is too short for memory to pay off, and the bottleneck is elsewhere.
- **The bottleneck is the verifier:** on Feb 6, both the stateless and memory systems accepted a harmful input-gain change, because the label-free score was fooled. Memory can't fix a wrong verdict it records as a success.
- **Next step:** let memory audit the verifier. Store each day's verifier score next to cheap delayed signals, such as later-session consistency or occasional user confirmation. Then learn which verifier scores actually predict good days, and down-weight agents whose "successes" don't hold up.

## Run
```bash
.venv/bin/python agentic/eval_memory_h1.py data/falcon_h1 agentic/out_memory
```
