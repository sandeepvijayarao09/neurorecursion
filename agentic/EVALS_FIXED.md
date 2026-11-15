# Older evals re-run after the embedding-cache fix

The earlier result files are unchanged. These re-runs are in new `*_fixed` folders.

| Eval | Before fix | After fix | Changed? |
|---|---|---|---|
| FALCON H1, frozen / agentic / no verifier / recal | −0.062 / −0.016 / −1.277 / +0.106 | identical | No (deterministic) |
| Memory agents, rule planner | −0.014 (worst −0.378) | −0.014 (worst −0.378) | No |
| Memory agents, Nemotron Super, single run | +0.013 (worst −0.090) | −0.020 (worst −0.378) | Yes, but from LLM sampling: a different single run |
| Partner comparison, rules | −0.014 | −0.014 | No |
| Partner: Nemotron 3 Super (3 runs) | −0.017 ± 0.008 | −0.003 ± 0.012 (worst −0.282) | Sampling noise |
| Partner: Nemotron 3.5 Lightning | −0.013 ± 0.001 | −0.013 ± 0.001 | No |
| Partner: Nemotron 3 Nano | −0.010 ± 0.004 | −0.013 ± 0.001 | Sampling noise |
| Partner: Nemotron 3.5 Super VL | −0.012 ± 0.001 | −0.013 ± 0.001 | Sampling noise |
| NVIDIA suite, FALCON ×3 seeds | −0.014 / −0.012 / −0.012 | −0.014 / −0.012 / −0.012 | No |
| NVIDIA suite, Version N+1 (MC_Maze) | 3 edits, all reverted | 3 different edits, all reverted | Different edits (sampling); same outcome |
| 14-session memory variants | see EVALS_14_SESSIONS.md | see EVALS_TRUTH_COMPARE.md | **Yes**: the split-half variants were affected |

**Conclusion:**
- **Where the bug mattered:** only the 14-session split-half variants, where the two verify halves are often the same length. Everything else is unchanged.
- **Where results moved:** only in LLM-planner runs, and only from run-to-run sampling. Nemotron 3 Super still lands within noise of the rule planner (mean −0.003 ± 0.012 vs −0.014).

**Folders:**
- `out_falcon_fixed/`, `out_memory_fixed/`, `out_memory_nemotron_fixed/`
- `out_partners_fixed/`, `out_nvidia_fixed/`
