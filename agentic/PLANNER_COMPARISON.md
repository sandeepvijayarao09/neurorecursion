# Does the choice of LLM planner matter? Four Nemotron models compared

**Setup:**
- **Task:** the memory-driven agent team (`eval_memory_h1.py`) on FALCON H1, 7 real held-out days in date order.
- **What varies:** only the planner model.
- **Runs:** 3 per model, with different seeds and temperature 0.6. The rule-based baseline is deterministic, so it ran once.
- **Fixed for every run:** frozen decoder R² −0.062, stateless agents −0.016.
- **Raw output:** `out_planners/planner_comparison.json`.

| Planner | Runs | Mean R² (± SD) | Range across runs | vs stateless | Worst day (mean) | Calls OK | Median latency |
|---|---|---|---|---|---|---|---|
| Rules (no LLM) | 1 | −0.014 | — | +0.001 | −0.378 | — | — |
| Nemotron 3 Super 120B | 3 | −0.017 ± 0.008 | −0.028 … −0.012 | −0.002 | −0.378 | 100% | 0.4 s |
| Nemotron 3.5 Lightning 30B | 3 | −0.013 ± 0.001 | −0.014 … −0.012 | +0.003 | −0.378 | 100% | 1.2 s |
| Nemotron 3 Nano 30B | 3 | **−0.010 ± 0.004** | −0.014 … −0.005 | +0.005 | −0.378 | 100% | 0.7 s |
| Nemotron 3.5 Super VL | 3 | −0.012 ± 0.001 | −0.014 … −0.012 | +0.003 | **−0.320** | 100% | 0.5 s |

## What this shows
- **All four Nemotron models work reliably as planners:** 100% valid JSON replies at 0.4–1.2 s median latency.
- **The choice of planner barely changes accuracy:** every planner lands between −0.017 and −0.010 R², within run-to-run noise of each other and of the rule baseline. The agents, verifier and memory drive the result, not the LLM.
- **Correction to an earlier number:** the single run reported earlier (Nemotron 3 Super, +0.013 R², worst day −0.090) **did not reproduce**. In these 3 new runs Super averages −0.017, and every run hit the −0.378 failure day. Across all 4 Super runs the mean is about −0.010. The earlier figure was one favorable sample, where the planner happened to skip the input-gain agent on Feb 6.
- **What does hold across runs:**
  - every agentic planner beats the frozen decoder (−0.062)
  - none closes the gap to few-shot recalibration (+0.106)
- **What would let an LLM planner add value:** inputs that predict when an agent will hurt, such as memory that audits past verifier verdicts. Today the planner sees the same label-free score that gets fooled on Feb 6.

## Reproduce
```bash
.venv/bin/python agentic/compare_planners.py data/falcon_h1 agentic/out_planners 3
```
