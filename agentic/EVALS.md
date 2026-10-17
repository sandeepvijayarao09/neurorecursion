# Evals: agentic adaptation vs existing approaches

## 1. Real cross-day drift: FALCON H1 (DANDI 000954)
- **Data:** human participant, intracortical arrays, 7-DoF reach and grasp.
- **Training:** the decoder is trained once on 6 held-in days (Jan 1–20), then frozen.
- **Testing:** 7 held-out days, 6–20 days later.
- **Adaptation:** agents use only the first 30% of each day's spikes, with no labels.
- **Scoring:** velocity R² on the remaining 70% of each day.

Decoder: causally smoothed rates (τ = 300 ms, 120 ms lag) with a ridge head. Held-in validation R² is 0.34.

| Condition | Labels needed on new day | Mean R² (95% CI) | Days better than frozen |
|---|---|---|---|
| Frozen decoder (no adaptation) | none | −0.062 (−0.121, −0.008) | — |
| **Agentic (with verifier)** | **none** | **−0.016 (−0.158, 0.092)** | **6 / 7** |
| Agentic, verifier off (ablation) | none | −1.277 (−1.745, −0.955) | 0 / 7 |
| Only channel-health agent | none | −0.013 | — |
| Only input-gain agent | none | +0.001 | — |
| Only latent-alignment agent | none | −0.062 (never accepted) | — |
| Few-shot recalibration (today's method) | ~3.5 min of labelled data per day | **+0.106 (0.049, 0.157)** | 7 / 7 |

Agentic minus frozen: +0.047 mean R² (95% CI −0.073 to +0.125). The median agentic system recovers 55% of the gain that recalibration gets, with zero labels.

**Honest read:**
- **Agents beat the frozen decoder on 6 of 7 real days,** without any labels.
- **On average they're not yet a replacement for recalibration.** Recalibration is still about 0.12 R² better, and the agentic CI overlaps zero gain.
- **The verifier is essential.** With it off, the same agents wreck the decoder (R² −1.28).
- **One failure day (+17 d, R² −0.38):** the verifier's label-free score (match of output distribution to day 0) accepted a harmful input-gain change. The next fix is a stricter verifier or a rollback rule.
- **I fixed one bug after seeing test results:** an agent could be applied twice. The first run, with that bug, is saved as `out_falcon/falcon_h1_eval_smooth_run1_bug.json`, mean R² −0.069. The fix matches the rule already in `recursive.py`.
- **Not run on this data:**
  - The intent agent was disabled, because H1 has no target positions.
  - The LLM planner and reviewer weren't used, because no `CRUSOE_API_KEY` was set.
- **Absolute R² is low for every method.** FALCON's published decoders score higher, so these numbers compare methods with each other, not against the benchmark leaderboard.

## 2. Simulated drift: MC_Maze (monkey, DANDI 000139)
The recursive loop ran for 4 versions with a rule-based reviewer:
- v0 held-out R² was 0.059, vs 0.035 for the frozen decoder.
- None of the 4 proposed playbook edits beat v0 on held-out scenarios.
- The gate reverted all 4 edits.

Results are in `out_recursive/scoreboard.json`.

## 3. Comparison with the existing system
| | Existing (daily recalibration) | NeuroRecursion (this work) |
|---|---|---|
| Labels per day | ~3.5 min few-shot (here); Neuralink reports ~55 min/week | 0 |
| Decoder changes | head refit daily | none (frozen) |
| Real-data R² (H1 mean) | **+0.106** | −0.016 |
| Beats frozen on | 7 / 7 days | 6 / 7 days |
| Safety mechanism | human | verifier + gate (without it: −1.28) |
| Audit trail | none | every proposal logged |

## Reproduce
```bash
MASTER=smooth .venv/bin/python agentic/eval_falcon_h1.py data/falcon_h1 agentic/out_falcon
```
