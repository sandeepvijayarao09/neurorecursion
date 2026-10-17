# NeuroRecursion: agents that keep a brain-computer interface working without recalibration

Recursive Agents Hackathon (Crusoe × Cerebral Valley), Oct 17 2026.

**Problem:** BCI decoders degrade every day. Electrodes die, channels get noisy, and neural signals drift. Today the fix is a daily recalibration session, about 55 min a week for Neuralink users, plus retraining the model.

**Our approach:** freeze the decoder and let a recursive agent team maintain everything around it. The team:
- writes its own definition of done for the day
- checks itself with a label-free verifier
- escalates to a human when safety checks fail
- rewrites its own playbook between versions, keeping a change only if it beats the previous version on drift scenarios it hasn't seen

## Tracks covered
| Track | Where it lives |
|---|---|
| **Version N+1** | `recursive.py: review()` + the held-out gate in `main()`. The reviewer agent rewrites the playbook; the change is kept only if held-out R² improves. See `out_recursive/scoreboard.json`. |
| **Done Means Done** | `plan_day()` turns the one-line goal into runnable checks (intent gain, speed range, masking limit). `check_dod()` runs them, and `decide_next()` chooses continue, re-plan, stop or escalate. |
| **Shared Brain** | `SharedBrain` in `memory.json`: lean notes plus per-agent stats. The planner reads it before each day and the reviewer updates it after each version. |

## Partner stack
- **Crusoe Managed Inference** (OpenAI-compatible endpoint `api.inference.crusoecloud.com/v1`):
  - **planner:** `nvidia/Nemotron-3-Super-120B-A12B`, for agentic reasoning
  - **reviewer:** `nvidia/nemotron-3.5-lightning-30b-a3b`, a cheap sub-agent
  - model IDs come from Crusoe's hackathon skill file; confirm them with `GET /v1/models`
- **NVIDIA:** both reasoning agents run on Nemotron. The tool agents are deterministic numerical code (channel masking, CORAL feature alignment, Procrustes rotation). The frozen decoder uses a Mamba-style state-space encoder, the same architecture family as Nemotron's hybrid Mamba design.
- **You.com (optional, not wired yet):** a research agent that pulls cited drift-mitigation methods when the verifier stalls.
- **NemoClaw / OpenShell (optional):** run `recursive.py` as an always-on agent in a sandbox.

## Run
```bash
cd neuralink_research
read -rs CRUSOE_API_KEY && export CRUSOE_API_KEY
.venv/bin/python agentic/recursive.py data/mc_maze/mc_maze_medium_train.nwb agentic/out_recursive 2
```
Without `CRUSOE_API_KEY` the planner and reviewer fall back to rules, so the pipeline still runs end to end.

## How it works
```
Version N playbook ──► for each day: Nemotron planner writes checks + agent order
                                   │
       spikes ─► channel-health / alignment / intent agents ─► verifier (label-free)
                                   │ accept / reject / iterate        │
                                   ▼                                  ▼
                       adaptation state around FROZEN decoder ─► cursor
                                   │
                    checks pass? ─► done | continue | re-plan | escalate to human
after all runs: Nemotron reviewer reads audit log + shared memory ─► playbook N+1
                gate: keep N+1 only if held-out R² (unseen drift scenarios) improves
```

## Data and what was built today
| Part | Source |
|---|---|
| Neural data | Public **MC_Maze** (DANDI 000139, monkey motor cortex) |
| Raw-signal stage | Neuralink's public 2024 compression-challenge recordings |
| Drift | Simulated: dead channels, noisy channels, per-channel gain changes |
| Built during the event | Everything under `agentic/` and `demo/` |

This is a research prototype, not a medical device, and it gives no medical advice.

## 1-minute demo script
1. **0:00–0:10, the problem:** decoders degrade daily; users recalibrate for about an hour a week.
2. **0:10–0:30:** run `recursive.py` live and show the planner's checks and the agent decisions streaming into `recursive_audit.jsonl`.
3. **0:30–0:45:** show `scoreboard.json`: v0 → v1 → v2 held-out R², and which edits were kept or reverted.
4. **0:45–0:60:** show `memory.json`, what the team learned, and the escalation path.

## Live Nemotron run (Crusoe Managed Inference)
Model IDs confirmed via `/v1/models`:
- **planner:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`
- **reviewer:** `nvidia/Nemotron-3.5-Lightning-30B-A3B`

Connection check: `python agentic/check_crusoe.py`.

**Version N+1 on MC_Maze** (`out_nemotron/`):
- 55 planner calls, median 1.9 s. 14 of them hit a 429 rate limit and were retried with backoff.
- 3 reviewer calls, median 0.7 s.
- The Nemotron reviewer proposed 3 playbook edits:
  - `dead_ratio` 0.35 + `verifier_margin` 0.02
  - `dead_ratio` 0.25 + `verifier_margin` 0.015
  - `dead_ratio` 0.25 + disable `alignment/input`
- The held-out gate reverted all 3 (none beat v0's 0.059). So the gate blocks unhelpful self-edits, but no improving edit has been found yet.

**Memory agents on real FALCON H1 days, with the Nemotron planner** (`out_memory_nemotron/`):

| | Frozen | Stateless agents | Memory + Nemotron planner |
|---|---|---|---|
| Mean R² (7 days) | −0.062 | −0.016 | **+0.013** |
| Worst day | −0.203 | −0.378 | **−0.090** |

- The planner skipped input gain on Feb 3 and Feb 6. That avoided the Feb 6 failure but gave up the Feb 3 gain.
- Memory minus stateless: +0.029 R² (95% CI −0.037 to +0.123), better on 3 days, worse on 2.
- This is a single run at temperature 0.6; repeated runs will vary.
