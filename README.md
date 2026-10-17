# NeuroRecursion

Recursive agents that keep a brain-computer-interface decoder usable **without daily recalibration**.

The decoder is trained once and frozen. A team of agents adapts the signal around it every day:
- channel-health, alignment and intent tool agents propose changes
- a label-free verifier accepts or rejects them
- a Nemotron planner on Crusoe writes each day's definition of done
- a Nemotron reviewer rewrites the playbook, which is kept only if it wins on held-out data
- a per-user memory layer (SQLite) learns what works for each person

Built at the Crusoe × Cerebral Valley **Recursive Agents Hackathon** (Oct 17, 2026). Research prototype; not a medical device and gives no medical advice.

## Repo map
| Path | What |
|---|---|
| `WORKFLOW.md`, `pipeline_data.json` | How Neuralink turns brain signals into outputs, from public sources |
| `demo/` | Stage A: raw Neuralink signal → spikes → compression. Stage B: spikes → pretrained encoder → cursor, with `RESULTS.md` |
| `agentic/agents.py` | Frozen-model wrapper, adaptation state, tool agents, verifier, orchestrator |
| `agentic/recursive.py` | Version N+1 loop: planner, definition-of-done checks, reviewer, held-out gate, shared brain |
| `agentic/memory.py`, `agentic/brain.py` | Per-user memory layer and memory-driven orchestrator (Thompson-sampled agent schedule, warm start, rollback) |
| `agentic/llm.py`, `agentic/check_crusoe.py` | Crusoe Managed Inference client (Nemotron 3 Super + Nemotron 3.5 Lightning) |
| `agentic/eval_falcon_h1.py`, `agentic/eval_memory_h1.py` | Evals on **real** cross-day human iBCI data (FALCON H1) |
| `agentic/EVALS.md`, `agentic/MEMORY_ARCHITECTURE.md`, `agentic/HACKATHON.md` | Results, architecture and demo script |
| `viz/` | Standalone 3D pages: existing system vs agent system, plus evals |

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/get_data.sh
mkdir -p ~/.config/crusoe && read -rs K && printf %s "$K" > ~/.config/crusoe/api_key && chmod 600 ~/.config/crusoe/api_key && unset K
.venv/bin/python agentic/check_crusoe.py
.venv/bin/python agentic/eval_memory_h1.py data/falcon_h1 agentic/out_memory_nemotron
```
Without a Crusoe key, the planner and reviewer fall back to rule-based policies.

## Headline result (FALCON H1, 7 real held-out days, mean velocity R²)
| Frozen decoder | Stateless agents | Memory agents + Nemotron planner | Few-shot recalibration (needs labels) |
|---|---|---|---|
| −0.062 | −0.016 | **+0.013** | +0.106 |

- Agents with memory plus the Nemotron planner beat the frozen decoder without any labels.
- Few-shot recalibration, which needs labels, still does better.
- All the numbers and caveats are in `agentic/EVALS.md` and `agentic/HACKATHON.md`.

## Data
All data is public and downloaded by `scripts/get_data.sh`; none of it is stored in this repo:
- MC_Maze (DANDI 000139)
- FALCON H1 (DANDI 000954)
- Neuralink's 2024 compression-challenge recordings, from a community mirror
