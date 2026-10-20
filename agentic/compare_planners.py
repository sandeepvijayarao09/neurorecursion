"""Compare planner models on Crusoe (and the rule baseline) on FALCON H1, several runs each.
Usage: python compare_planners.py <falcon_h1_dir> <out_dir> [runs]"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MODELS = ["rules", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B", "nvidia/Nemotron-3.5-Lightning-30B-A3B",
          "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/NVIDIA-Nemotron-3.5-Super-VL"]


def run(model, seed, root, out):
    d = os.path.join(out, model.split("/")[-1], f"seed{seed}")
    env = {**os.environ, "SEED": str(seed)}
    if model == "rules":
        env["NO_LLM"] = "1"
    else:
        env["PLANNER_MODEL"] = model
    here = os.path.dirname(os.path.abspath(__file__))
    p = subprocess.run([sys.executable, os.path.join(here, "eval_memory_h1.py"), root, d], env=env,
                       capture_output=True, text=True)
    if p.returncode:
        return model, seed, {"error": p.stderr[-400:]}
    return model, seed, json.load(open(os.path.join(d, "memory_eval.json")))["summary"]


def main(root, out, runs=3):
    jobs = [(m, s) for m in MODELS for s in range(runs if m != "rules" else 1)]
    with ThreadPoolExecutor(len(MODELS)) as ex:          # one worker per model keeps per-model rate limits safe-ish
        res = list(ex.map(lambda j: run(j[0], j[1], root, out), jobs))
    table = {}
    for m, s, r in res:
        table.setdefault(m, []).append(r)
    rows = []
    for m, rs in table.items():
        ok = [r for r in rs if "error" not in r]
        if not ok:
            rows.append({"planner": m, "error": rs[0]["error"]})
            continue
        mem = np.array([r["memory"]["mean_r2"] for r in ok])
        worst = np.array([r["memory"]["worst_day"] for r in ok])
        rows.append({"planner": m, "runs": len(ok), "mean_r2": float(mem.mean()), "sd": float(mem.std()),
                     "min_run": float(mem.min()), "max_run": float(mem.max()), "worst_day_mean": float(worst.mean()),
                     "vs_stateless": float(np.mean([r["memory_minus_stateless"]["mean"] for r in ok])),
                     "llm_ok_rate": float(np.mean([r["llm"]["ok"] / max(1, r["llm"]["calls"]) for r in ok])) if m != "rules" else None,
                     "median_latency_s": float(np.median([r["llm"]["median_s"] or 0 for r in ok])) if m != "rules" else None,
                     "frozen": ok[0]["frozen"]["mean_r2"], "stateless": ok[0]["stateless"]["mean_r2"]})
    json.dump(rows, open(os.path.join(out, "planner_comparison.json"), "w"), indent=2)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
