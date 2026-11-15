"""Build agent_implementation.html: the agent system as actually implemented and evaluated,
with the model or algorithm named at every step (labelled agents only).

python build_implementation.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, "template.html")).read()

PINK, CLEAN, SIG, AMBER, GRAY, ORANGE = 0xED93B1, 0x85B7EB, 0x378ADD, 0xBA7517, 0x888780, 0xD85A30
GREEN, VIOLET, INDIGO = 0x1D9E75, 0x7F77DD, 0x534AB7

cfg = {
    "title": "Agent system · implementation",
    "subtitle": "What runs at each step and which model it uses",
    "nav": [{"label": "Existing system", "href": "existing_system.html", "on": False},
            {"label": "Agent system", "href": "agent_system.html", "on": False},
            {"label": "Implementation", "href": "agent_implementation.html", "on": True},
            {"label": "Evals", "href": "evals.html", "on": False}],
    "brainX": -18, "overview": {"r": 56, "t": [8, -.3, .8]}, "pulse": 3,
    "flow": "spikes → <b class='hl'>labelled agents + user memory</b> → frozen decoder → 7-DoF velocity",
    "stages": [
        {"name": "Brain", "sub": "motor-cortex intent", "short": "Human iBCI participant (FALCON H1)", "color": PINK,
         "detail": "A person with intracortical arrays attempts to follow a virtual arm in a 7-DoF reach-and-grasp task. "
                   "Their recorded intended velocity is the truth every system is scored against.",
         "kv": [["Dataset", "FALCON H1 · DANDI 000954"], ["Truth", "OpenLoopKinematicsVelocity"], ["Channels", "176"]]},
        {"name": "Cleaner", "sub": "raw → clean signal", "short": "Band-pass + threshold front end", "color": CLEAN,
         "shape": "c", "x": -11,
         "detail": "Built and measured on Neuralink's raw 2024 recordings. FALCON H1 arrives already cleaned, as "
                   "threshold crossings, so this step is upstream of the evals.",
         "subs": [["Band-pass", "300–6000 Hz Butterworth", "c"], ["Threshold", "−4.5σ MAD noise", "c"],
                  ["Refractory", "1 ms", "c"]],
         "kv": [["Model", "none (DSP)"], ["Labels", "no"]]},
        {"name": "Spike counts", "sub": "20 ms bins", "short": "Spikes binned and smoothed", "color": SIG, "shape": "c",
         "x": -2,
         "detail": "Spike events per channel are counted in 20 ms bins, then causally smoothed. This is the input "
                   "the agents and decoder see.",
         "subs": [["Bin", "20 ms", "c"], ["Smooth", "causal τ = 300 ms", "c"], ["Compression", "≈1,600× vs raw", "c"]],
         "kv": [["Model", "none (DSP)"], ["Labels", "no"]]},
        {"name": "Labelled agents", "sub": "~14 s labels per session", "short": "Agents checked against labels",
         "color": VIOLET, "shape": "b", "big": 1.12, "x": 10,
         "detail": "Agents propose small reversible changes using the session's calibration labels. A labelled "
                   "verifier accepts a change only if R² improves on both held-back halves, and an output adapter "
                   "fits a 7×7 correction anchored to training data. Memory starts each session from the last "
                   "verified state. The hybrid runs these agents on top of the recalibrated head.",
         "subs": [["User memory", "SQLite · last verified state", "c", AMBER], ["Calibration labels", "first 30% · ~14 s", "c", AMBER],
                  ["Health / align", "rule + statistics agents", "b"], ["Labelled verifier", "split-half R²", "s"],
                  ["Output adapter", "7×7 ridge, anchored", "b"]],
         "kv": [["Planner", "rule-based (no LLM in this eval)"], ["Labels", "same budget as current system"],
                ["Test R², label agents + memory", "+0.002"], ["Test R², hybrid", "+0.093 (current +0.087)"]]},
        {"name": "Frozen decoder", "sub": "trained once on Jan 1–20", "short": "Ridge on smoothed rates, never retrained",
         "color": GRAY, "shape": "model", "slab": GRAY, "x": 22, "flow": GRAY,
         "detail": "On FALCON H1 the frozen decoder is a ridge regression on causally smoothed spike rates "
                   "(120 ms lag). On MC_Maze it is a Mamba-style state-space encoder pretrained with masked "
                   "Poisson prediction. Neither is a multi-user foundation model yet.",
         "subs": [["Features", "smoothed rates", "b"], ["Head", "ridge, 7 outputs", "b"], ["Held-in R²", "0.34", "b"]],
         "kv": [["FALCON H1 model", "smoothed-rate ridge"], ["MC_Maze model", "Mamba-style SSM encoder"],
                ["Weights updated", "0%"]]},
        {"name": "Outputs", "sub": "7-DoF velocity", "short": "Reach, wrist and grasp velocity", "color": ORANGE,
         "shape": "s", "x": 31,
         "detail": "Decoded velocity for reach x/y/z, wrist rotation and three grasp dimensions, scored against "
                   "the truth on the last 70% of each session.",
         "subs": [["Reach", "x · y · z", "s"], ["Wrist", "rotation", "s"], ["Grasp", "3 dims", "s"]],
         "kv": [["Metric", "velocity R²"], ["Scored on", "last 70% of each session"]]},
    ],
    "loop": {"from": 5, "to": [3, 0], "up": False, "color": AMBER, "stage": 3, "speed": .14, "z": 1.5,
             "label": "User memory", "sub": "verdicts + states carried to the next session"},
    "legend": [["Signal path", SIG], ["Labelled agents", VIOLET], ["User memory", AMBER],
               ["Frozen decoder", GRAY], ["Outputs", ORANGE]],
    "metrics": [["Label agents + memory", "+0.002 R²", "warn"],
                ["Hybrid vs current", "+0.093 vs +0.087", "good"], ["Frozen, no help", "−0.100 R²", "bad"]],
    "overviewCard": {"name": "How the implementation works",
                     "detail": "The front end turns brain activity into spike counts. Labelled agents with user memory "
                               "adapt them for a decoder that is never retrained, using the same ~14 s of calibration "
                               "labels the current system needs.",
                     "kv": [["Data", "FALCON H1, 14 real sessions"], ["Tuning / unseen", "sessions 1–7 / 8–14"]]},
    "footnote": "Test R² is the mean over unseen sessions 8–14 (agentic/EVALS_LABELED_AGENTS.md). "
                "The labelled-agent eval used the rule-based planner, not an LLM.",
}

html = (TPL.replace("__TITLE__", cfg["title"]).replace("__ACCENT__", "#0F6E56")
        .replace("__ACCENT_BG__", "#E1F5EE").replace("__ACCENT_BG_DARK__", "rgba(29,158,117,.18)")
        .replace("__CONFIG__", json.dumps(cfg, ensure_ascii=False)))
out = os.path.join(HERE, "agent_implementation.html")
open(out, "w").write(html)
print("wrote", out)
