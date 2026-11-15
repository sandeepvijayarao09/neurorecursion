# Truth vs agent system vs current system (FALCON H1, 14 sessions)

- **Truth:** recorded 7-DoF velocity.
- **Agent system:** frozen decoder plus memory agents, with no labels on new sessions. This is corrected v3: split-half check, delayed credit, and a warm start that must pass the split-half check.
- **Current system:** decoder refit every session from that session's calibration labels.

Scoring uses the last 70% of each session.

| | Mean R² | Mean correlation r | RMSE |
|---|---|---|---|
| Agent system (no labels) | **−0.010** | 0.396 | 0.00398 |
| Current system (labels) | **+0.050** | 0.400 | 0.00392 |
| Frozen, no help | −0.158 | 0.394 | 0.00416 |

**Findings:**
- **Shape is the same for all three:** every system tracks the shape of the movement about equally (r ≈ 0.39–0.40). The differences in R² come from size and offset errors.
- **What agents fix:** they correct most of the frozen decoder's size and offset error (R² −0.158 → −0.010) without labels.
- **Current system still leads overall:** agents are closer to truth in 4 of 14 sessions.

## Correction: embedding cache bug (found during this eval)
`FrozenMaster.latent()` cached embeddings by Python `id()` of the input array. A freed array's id can be reused by a new array of the same length, which returned a stale embedding. This mainly affected the split-half check, whose two halves are often the same length.

The fix keeps a reference to the cached array (`agents.py`). Every 14-session variant was then re-run into a **new** folder, `out_14sessions_fixed/`. Earlier result files were not changed.

| Variant | Dev R² before fix | Dev R² after fix | Test R² after fix |
|---|---|---|---|
| v3_split_warmsplit (previous pick) | −0.005 | **−0.050** | −0.019 |
| v3_split_delay_warmsplit (corrected pick, stable over 3 seeds) | −0.029 | **−0.001** | −0.019 |
| v3_split_nowarm | −0.004 | −0.016 / −0.013 | −0.019 |
| v1_memory | −0.049 | −0.049 | −0.011 |

**What changes:**
- **The previous v3 pick only looked good because of the bug.** After the fix, the dev-selected system is split-half + delayed credit + checked warm start.
- **Test is unchanged:** all corrected v3 variants score −0.019 on test.
- **v1 has the best test score** (−0.011), so the memory-layer changes still don't show a test gain.

**Other evals not yet re-run after the fix:**
- `EVALS.md`, `MEMORY_ARCHITECTURE.md`, `PARTNER_EVALS.md`, `NVIDIA_EVALS.md`, `EVALS_14_SESSIONS.md`
- Treat their split-half and memory numbers as possibly affected until they're re-run.
- The FALCON evals with the smoothed decoder call the verifier on different-length slices, so they're less exposed, but this is unverified.

## Files
- **Code:** `eval_truth_compare.py`
- **Data:** `out_truth/truth_compare.json`
- **Page:** `slides/truth_vs_agents.html`
- **Corrected reruns:** `out_14sessions_fixed/`
