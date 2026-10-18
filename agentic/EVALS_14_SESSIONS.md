# 14-session evals and memory-layer improvements (FALCON H1)

Each of the 14 held-out FALCON H1 recording sessions (Jan 26 – Feb 9, two per day) is treated as one "day". Every variant processes all 14 in order and carries its memory forward. The data is real, not simulated.

**Protocol:**
- Sessions 1–7 are **dev**: the only data used to decide which change to keep.
- Sessions 8–14 are **test**: reported, never used for decisions.
- Labels only score the last 70% of each session.
- The decoder is frozen, trained on the held-in days (Jan 1–20).
- Iteration rounds use the rule planner, so they're deterministic. The final run uses Nemotron 3 Super on Crusoe.

## Mean velocity R²
| System | Dev (1–7) | **Test (8–14)** | All 14 | Worst session | Beats frozen |
|---|---|---|---|---|---|
| Frozen decoder | −0.216 | −0.100 | −0.158 | −0.496 | — |
| Stateless agents | −0.018 | −0.028 | −0.023 | −0.293 | — |
| Few-shot recalibration (needs labels) | +0.014 | **+0.087** | +0.050 | −0.147 | — |
| v1 memory (previous) | −0.049 | −0.011 | −0.030 | −0.444 | 11/14 |
| **v3 memory, final (rules)** | **−0.005** | −0.015 | **−0.010** | **−0.292** | 11/14 |
| v3 memory + Nemotron planner (3 runs) | −0.005 | −0.020 (−0.017 … −0.021) | −0.012 | −0.292 | 9–10/14 |

## What was tried, and what was kept (dev only)
| Round | Change | Dev R² | Decision |
|---|---|---|---|
| 1 | v1 baseline | −0.049 | — |
| 1 | **Split-half check:** accept only if both halves of the verify slice improve | −0.033 | **keep** |
| 1 | Delayed credit: next session re-checks yesterday's changes | −0.049 | drop, no dev gain |
| 1 | Memory-learned gain bounds | −0.084 | drop, blocks good fixes |
| 2 | Split-half + no warm start | −0.004, −0.017, −0.004 (3 seeds) | runner-up |
| 2 | **Split-half + warm start that must also pass the split-half check** | −0.005 on all 3 seeds | **keep (v3)** |
| 2 | Split-half + delayed credit + checked warm start | −0.029 | drop |

The change in round 2 came from a specific failure. In session 4, v1 warm-started from memory and then stacked another gain change on top, which drove R² to −0.444. v3 trusts a remembered state only if it wins on both halves of today's data.

## Honest read
- **Clear gains on dev:** v3 lifts dev R² from −0.049 to −0.005, and cuts the worst session from −0.444 to −0.292.
- **Gains don't carry fully to unseen sessions:**
  - v3 test (−0.015) is better than stateless agents (−0.028) and far better than frozen (−0.100)
  - but it's slightly behind v1 on test (−0.011)
  - the dev-selected changes mostly made the system safer, not more accurate, on new data
- **Nemotron doesn't help on top of v3 yet:** 42 calls, all valid. Test R² was −0.020 against −0.015 with the rule planner, within noise.
- **Recalibration still leads on test** (+0.087). The agents close about 77% of the frozen-to-recalibration gap with zero labels: (−0.015 − (−0.100)) / (0.087 − (−0.100)).
- **Still open:** delayed credit didn't help with sessions this short (about 15 s of adaptation data each). It may need longer sessions, or a stronger delayed signal.

## Files (all new; earlier evals untouched)
- **Code:** `brain_v2.py` (the safeguards) and `eval_14_sessions.py` (14-session harness)
- **Results:** `out_14sessions/`
  - `eval_14_sessions.json`: round 1
  - `eval_14_round2*.json`: round 2 and seed tie-break
  - `eval_14_final_nemotron_seed*.json`: final Nemotron runs
  - `crusoe_calls_seed*.jsonl`: Crusoe call logs

```bash
NO_LLM=1 .venv/bin/python agentic/eval_14_sessions.py data/falcon_h1 agentic/out_14sessions
SEED=0 OUT_NAME=final.json .venv/bin/python agentic/eval_14_sessions.py data/falcon_h1 agentic/out_14sessions v3_split_warmsplit
```
