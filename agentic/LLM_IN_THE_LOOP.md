# Putting an LLM in the loop: Nemotron planner and reviewer

Run on Oct 17 2026. Every LLM decision came from a live Crusoe call. The request ID, token counts and reply of each call are logged in `out_nvidia/**/crusoe_calls.jsonl`.

## 1. Version N+1 self-improvement loop (MC_Maze, simulated drift)
- **Models:** Nemotron 3 Super 120B plans every day. Nemotron 3.5 Lightning reviews and rewrites the playbook between versions.
- **Rounds:** 3 improvement rounds.
- **Gate:** keep a version only if held-out R² improves.

| Version | Proposed by | Edit | Held-out R² | Kept |
|---|---|---|---|---|
| v0 | — | starting playbook | 0.059 | ✓ |
| round 1 | Nemotron Lightning | `verifier_margin` 0.03, `max_mask_frac` 0.4 | 0.043 | reverted |
| round 2 | Nemotron Lightning | `dead_ratio` 0.15, `max_mask_frac` 0.4 | 0.059 | reverted (tie) |
| round 3 | Nemotron Lightning | same as round 2 | 0.059 | reverted (tie) |

Frozen decoder on the same held-out scenarios: 0.035.

**Findings:**
- **The gate worked:** it blocked one harmful edit (round 1) and two that made no difference.
- **The reviewer repeated itself:** it re-proposed round 2's edit in round 3, even though reverted edits were passed to it as "do not repeat". A hard filter on reverted edits is needed, not just a prompt instruction.

## 2. Memory-driven agents on real cross-day human data (FALCON H1)
- **Planner:** Nemotron 3 Super 120B orders the agents each day from the user's memory.
- **Data:** 7 held-out days, processed in date order.
- **Runs:** 3 seeds.

| | Frozen | Stateless agents | Memory + Nemotron (3 runs) |
|---|---|---|---|
| Mean R² | −0.062 | −0.016 | **−0.013 ± 0.001** (−0.014, −0.012, −0.012) |
| vs stateless | | | +0.001 to +0.004 |
| Worst day | −0.203 | −0.378 | −0.378 in all 3 runs |

**Findings:**
- **Consistent across runs:** all 3 beat the frozen decoder by about 0.05 R², with small spread between runs.
- **No real gain over stateless agents:** the edge is +0.001 to +0.004.
- **The planner didn't prevent the Feb 6 failure:** it kept scheduling the input-gain agent. Sometimes it returned an empty plan, and the system then fell back to the memory-based schedule.

## Calls, latency and cost
| Suite | Model | Calls | Valid JSON | Median latency | Tokens in / out | Cost |
|---|---|---|---|---|---|---|
| Version N+1 | NVIDIA-Nemotron-3-Super-120B-A12B | 42 | 42 / 42 | 0.9 s | 24,999 / 3,096 | $0.0149 |
| Version N+1 | Nemotron-3.5-Lightning-30B-A3B | 3 | 3 / 3 | 0.7 s | 2,138 / 320 | $0.0002 |
| FALCON H1 ×3 | NVIDIA-Nemotron-3-Super-120B-A12B | 21 | 21 / 21 | 0.4 s | 3,829 / 718 | $0.0029 |
| **Total** | | **66** | **100%** | | **30,966 / 4,134** | **$0.018** |

Cost uses Crusoe's listed prices: Super $0.30 / $2.40 and Lightning $0.05 / $0.20 per 1M tokens.

**Example request IDs:**
- Super: `chatcmpl-a9d58ffdde98e890`, `chatcmpl-80d1b3a46ce642bb`
- Lightning: `chatcmpl-8aa604ed`

## Reproduce
```bash
LLM_LOG=agentic/out_nvidia/recursive/crusoe_calls.jsonl .venv/bin/python agentic/recursive.py data/mc_maze/mc_maze_medium_train.nwb agentic/out_nvidia/recursive 3
for s in 0 1 2; do SEED=$s LLM_LOG=agentic/out_nvidia/falcon_h1/seed$s/crusoe_calls.jsonl .venv/bin/python agentic/eval_memory_h1.py data/falcon_h1 agentic/out_nvidia/falcon_h1/seed$s; done
```
