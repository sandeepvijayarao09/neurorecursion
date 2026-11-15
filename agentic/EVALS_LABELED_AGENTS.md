# Agents that use labels (FALCON H1, 14 sessions)

Every labelled method gets the **same label budget** as the current system: labels on the first 30% of each session (about 14 s). Scoring is on the last 70%. Dev is sessions 1–7, used for decisions; test is sessions 8–14, reported only.

**Methods:**
- **Current system:** refit the decoder head with the session's labels, pooled with the training data.
- **Label agents:** the decoder stays frozen. Agents (channel health, gain and latent alignment, plus a new **output adapter**, a 7×7 correction fit from labels) are accepted only if a **labelled verifier** says R² improves.
- **Label agents + memory:** the same, but a session starts from the last verified state when that scores better.
- **Hybrid:** the current system's refit head, with label agents and memory around it.

## Results, mean velocity R²
| System | Labels? | Dev | **Test** | All 14 | Worst | vs current (95% CI) |
|---|---|---|---|---|---|---|
| Frozen | none | −0.216 | −0.100 | −0.158 | −0.496 | |
| Label-free agents (v3, fixed) | none | −0.001 | −0.019 | −0.010 | −0.291 | |
| **Current system** | slice | +0.014 | +0.087 | +0.050 | −0.147 | — |
| Label agents, round 1 | slice | −0.043 | −0.169 | −0.106 | −0.621 | −0.156 (−0.250, −0.075) |
| Label agents, round 2 | slice | −0.050 | −0.013 | −0.031 | −0.265 | −0.082 (−0.125, −0.031) |
| Label agents + memory, round 2 | slice | −0.015 | **+0.002** | −0.007 | −0.258 | −0.057 (−0.106, −0.005) |
| **Hybrid, round 2** | slice | **+0.014** | **+0.093** | **+0.054** | −0.147 | **+0.003 (−0.005, +0.014)** |

## What changed between rounds
- **Round 1 failed.** The labelled verifier scored changes on only about 6 s of labels. That's noisy enough to approve harmful changes: a −0.621 session, where an adapter fit on about 9 s of labels overfit.
- **Round 2 fixes:**
  1. The labelled verifier must show improvement on **both halves** of the labelled slice.
  2. The output adapter is **anchored**, fit on the session's labels pooled with about 6.8k training-day pairs, the same trick that makes the current system robust.

## Honest read
- **Labels help agents only when used carefully.** Naive use is worse than no labels at all (−0.169 vs −0.019 on test). With the two round-2 safeguards, label agents with memory reach +0.002 on test. That beats label-free agents (−0.019) and frozen (−0.100), but is still below the current system (+0.087).
- **The hybrid matches the current system without degrading it:** test +0.093 vs +0.087, difference +0.003 with a CI that includes 0. Agents on top of the refit decoder don't hurt, and add a small, not-significant gain.
- **The only setup that beats the current system on dev and test is the hybrid,** and only by a statistical tie. The case for agents is still adaptation with zero or few labels, not beating full recalibration.
- **Decisions came from dev only:** round 2 was designed after round 1's dev failures, and both rounds' test numbers are shown.

## Files (new)
- **Code:** `eval_labeled_agents.py` (round 1) and `eval_labeled_agents_v2.py` (round 2)
- **Results:** `out_labeled/labeled_agents.json` and `out_labeled/labeled_agents_v2.json`
- **Related:** `EVALS_FIXED.md` covers the cache-fix re-runs of the older evals
