"""Build viz/evals.html from the saved evaluation outputs.

python build_evals.py   (re-run after any experiment to refresh the page)
"""
import collections
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
J = lambda p: json.load(open(os.path.join(ROOT, p)))

A = J("demo/out/stage_a_summary.json")
CV = J("demo/out/stage_b_crossval.json")
AG = J("agentic/out/agentic_results.json")
SB = J("agentic/out_recursive/scoreboard.json")
dec = collections.Counter()
for line in open(os.path.join(ROOT, "agentic/out_recursive/recursive_audit.jsonl")):
    r = json.loads(line)
    if "decision" in r:
        dec[(r["agent"], r["decision"])] += 1
agents = ["channel_health", "alignment/input", "alignment/latent", "intent"]

data = {
    "generated": time.strftime("%Y-%m-%d %H:%M"),
    "A": {"labels": ["Raw 10-bit", "Best lossless", "20 ms counts", "Spike events"],
          "mbps": [A["N1_1024ch_Mbps"][k] for k in ["raw_10bit", "lossless_lzma", "binned_counts_20ms", "spike_events"]],
          "ratio": [A["compression_vs_raw"][k] for k in ["raw_10bit", "lossless_lzma", "binned_counts_20ms", "spike_events"]],
          "spikes": A["total_spikes"], "minutes": A["recording_minutes"], "files": A["files"],
          "rate_mean": A["rate_hz"]["mean"], "rate_median": A["rate_hz"]["median"]},
    "B": {"fracs": list(CV["label_efficiency_mean_r2"].keys()),
          "raw": [v["raw"] for v in CV["label_efficiency_mean_r2"].values()],
          "emb": [v["emb"] for v in CV["label_efficiency_mean_r2"].values()],
          "folds": [[f["fold"], f["r2_all"]["raw"], f["r2_all"]["emb"], f["r2_movement"]["raw"], f["r2_movement"]["emb"],
                     f["acc"]["true"], f["acc"]["raw"], f["acc"]["emb"]] for f in CV["per_fold"]],
          "sel": CV["selection"], "diff_ci": CV["accuracy_diff_emb_minus_raw_95ci"],
          "r2_all": CV["r2_all_mean"], "r2_move": CV["r2_movement_mean"], "trials": CV["trials_scored"], "targets": CV["n_targets"]},
    "C": [[r["day"], r["frozen"]["r2"], r["agentic"]["r2"], r["recal"]["r2"], r["agent_state"]["masked_channels"]] for r in AG],
    "D": {"board": SB["scoreboard"], "runtime": SB.get("runtime_s"), "llm_calls": len(SB.get("llm_calls", [])),
          "llm_ok": sum(1 for c in SB.get("llm_calls", []) if c[3] is True),
          "models": sorted({c[1] for c in SB.get("llm_calls", [])}),
          "rate_limited": sum(1 for c in SB.get("llm_calls", []) if "RateLimit" in str(c[3])),
          "agents": agents, "dec": {a: [dec[(a, "accept")], dec[(a, "iterate")], dec[(a, "reject")]] for a in agents}},
}

TPL = open(os.path.join(HERE, "evals_template.html")).read()
open(os.path.join(HERE, "evals.html"), "w").write(TPL.replace("__DATA__", json.dumps(data)))
print("wrote", os.path.join(HERE, "evals.html"))
