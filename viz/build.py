"""Generate the two standalone 3D pipeline pages from template.html.

python build.py   ->  existing_system.html, agent_system.html
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, "template.html")).read()

PINK, CLEAN, SIG, RED, AMBER, GRAY, ORANGE = 0xED93B1, 0x85B7EB, 0x378ADD, 0xE24B4A, 0xBA7517, 0x888780, 0xD85A30
GREEN, VIOLET, INDIGO = 0x1D9E75, 0x7F77DD, 0x534AB7
XS = [-11, -2, 9, 20, 29]

BRAIN = {"name": "Brain", "sub": "motor-cortex intent", "short": "Neurons fire when the user intends to move",
         "color": PINK,
         "detail": "Electrodes sit in the hand area of motor cortex. When the user intends to move, nearby neurons fire. "
                   "That electrical activity is the raw input to every downstream step.",
         "kv": [["Electrodes", "1,024 on 64 threads"], ["Sampling", "~20 kHz, 10-bit"], ["Raw data rate", "≈200 Mbps"]]}
CLEANER = {"name": "Cleaner", "sub": "raw → clean signal", "short": "Filter noise, re-reference, reject artifacts",
           "color": CLEAN, "shape": "c", "x": XS[0],
           "detail": "Removes slow drift and high-frequency noise, subtracts signals common to all channels, and drops "
                     "segments corrupted by motion or clipping.",
           "subs": [["Band-pass", "300–6000 Hz", "c"], ["Re-reference", "common average", "c"], ["Artifact reject", "motion / clipping", "c"]]}
SPIKES = {"name": "Spike detection", "sub": "clean → spike events", "short": "Threshold crossings become compact events",
          "color": SIG, "shape": "c", "x": XS[1],
          "detail": "Each threshold crossing becomes a timestamped event per channel. This shrinks the data roughly "
                    "1,600× so it fits the implant's ~1 Mbps radio.",
          "subs": [["Threshold", "−4.5σ noise", "c"], ["Spike events", "time + channel", "c"], ["Radio", "≈0.12 Mbps used", "c"]],
          "kv": [["Compression vs raw", "≈1,600×"], ["Measured on", "Neuralink 2024 data"]]}
OUTPUTS = {"name": "Outputs", "sub": "cursor · clicks · typing", "short": "What the user actually controls",
           "color": ORANGE, "shape": "s", "x": XS[4],
           "detail": "Decoded intent drives the cursor, distinguishes click types and powers on-screen or ten-finger typing.",
           "subs": [["Cursor", "velocity", "s"], ["Clicks", "left / right", "s"], ["Typing", "on-screen / 10-finger", "s"]]}
NAV = lambda on: [{"label": "Existing system", "href": "existing_system.html", "on": on == "existing"},
                  {"label": "Agent system", "href": "agent_system.html", "on": on == "agent"},
                  {"label": "Evals", "href": "evals.html", "on": False}]

existing = {
    "title": "Existing BCI system",
    "subtitle": "Personalisation by retraining a small user head on a 99%-frozen foundation model",
    "nav": NAV("existing"), "brainX": -18, "overview": {"r": 60, "t": [5.5, .6, 0]}, "pulse": 3,
    "flow": "raw → clean → spikes → <b class='hl'>retrain user head (1%)</b> → foundation model <b>(99% frozen)</b> → outputs",
    "stages": [BRAIN, CLEANER, SPIKES,
               {"name": "User head retrain", "sub": "1% of weights · needs labels", "short": "Daily calibration refits the user-specific head",
                "color": RED, "shape": "b", "x": XS[2], "subsAbove": True,
                "detail": "Neural signals drift day to day, so control degrades. The user stops, performs a structured "
                          "calibration task, and the resulting labels are used to refit a small user-specific head. "
                          "Previous labels are discarded.",
                "subs": [["Drift detected", "control drops", "s"], ["Calibration task", "user · ~10 min/day", "h"],
                         ["New labels", "old ones discarded", "b", AMBER], ["Retrain head", "redeploy", "b", GRAY]],
                "kv": [["User time", "~55 min / week"], ["Repeated for", "every feature"]]},
               {"name": "Foundation model", "sub": "99% frozen", "short": "Shared encoder; only the user head changes",
                "color": GRAY, "shape": "model", "slab": RED, "x": XS[3], "flow": GRAY,
                "detail": "A pretrained spike encoder shared across users. Most weights stay frozen; the red top layer is "
                          "the user head that gets retrained after each calibration.",
                "subs": [["Spike tokens", "frozen", "b"], ["Encoder", "frozen", "b"], ["User head", "1% · retrained", "b", RED]]},
               OUTPUTS],
    "loop": {"from": 5, "to": [3, 0], "up": True, "color": RED, "stage": 3, "speed": .18, "z": -1,
             "label": "Daily recalibration", "sub": "drift → user → labels → retrain"},
    "legend": [["Signal path", SIG], ["Frozen model", GRAY], ["User in the loop", RED], ["Outputs", ORANGE]],
    "metrics": [["Calibration time", "~55 min / week", "bad"], ["Weights updated", "1%, daily", "warn"],
                ["Labels required", "Every session", "bad"], ["Old labels", "Discarded", "bad"]],
    "overviewCard": {"name": "How it works today",
                     "detail": "One shared front end turns brain activity into spikes. A foundation model decodes them, "
                               "but each user's signals drift, so a small user head is retrained daily from fresh "
                               "calibration labels. The user does the maintenance work.",
                     "kv": [["Who fixes drift", "The user"], ["Model changes", "Daily, per feature"]]},
    "footnote": "Calibration figures from Neuralink, “Pretraining on 50,000 Hours of Unlabeled Brain Data” (Oct 2026). "
                "“99% frozen” reflects Neuralink's cross-participant transfer with 99% of weights frozen.",
}

agent = {
    "title": "Agent system · NeuroRecursion",
    "subtitle": "Personalisation by translation: agents fit each user's signals to a 100%-frozen foundation model",
    "nav": NAV("agent"), "brainX": -18, "overview": {"r": 60, "t": [5.5, .2, .6]}, "pulse": 3,
    "flow": "raw → clean → spikes → <b class='hl'>agent translation layer + long-term memory</b> → foundation model <b>(100% frozen)</b> → outputs",
    "stages": [BRAIN, CLEANER, SPIKES,
               {"name": "Agent translation layer", "sub": "personal → foundation space", "short": "Agents adapt the input, not the model",
                "color": GREEN, "shape": "b", "big": 1.18, "x": XS[2],
                "detail": "A team of agents uses long-term memory of this user's signals to translate today's spikes "
                          "into the space the frozen foundation model expects. Every proposed change is small, "
                          "reversible and gated by a label-free verifier.",
                "subs": [["User memory", "months of history", "c", AMBER], ["Channel health", "mask / reweight", "b"],
                         ["Translate", "align to model space", "b"], ["Intent check", "task pseudo-labels", "b"],
                         ["Verifier", "accept / reject", "s", VIOLET]],
                "overhead": {"name": "Planner + reviewer", "sub": "Nemotron on Crusoe", "color": INDIGO},
                "kv": [["Labels used", "None"], ["Planner", "Nemotron 3 Super"], ["Reviewer", "Nemotron 3.5 Lightning"],
                       ["Escalates to human", "Only on failed safety checks"]]},
               {"name": "Foundation model", "sub": "100% frozen", "short": "Trained once, never modified",
                "color": GRAY, "shape": "model", "slab": GRAY, "x": XS[3], "flow": GRAY,
                "detail": "The same pretrained encoder and head for every user, fully frozen. All personalisation lives "
                          "in the agent layer in front of it, so the model never needs retraining.",
                "subs": [["Spike tokens", "frozen", "b"], ["Encoder", "frozen", "b"], ["Head", "frozen", "b"]]},
               OUTPUTS],
    "loop": {"from": 5, "to": [3, 0], "up": False, "color": AMBER, "stage": 3, "speed": .16, "z": 1.5,
             "label": "Long-term user memory", "sub": "outputs + verdicts feed memory"},
    "legend": [["Signal path", SIG], ["Agents", GREEN], ["Nemotron", INDIGO], ["Frozen model", GRAY], ["Outputs", ORANGE]],
    "metrics": [["Labels needed", "None (goal)", "good"], ["Weights updated", "0% · frozen", "good"],
                ["Personalisation", "Agents + memory", "good"], ["Safety", "Verifier + human", "good"]],
    "overviewCard": {"name": "How the agent system works",
                     "detail": "The same front end produces spikes. Instead of retraining weights, an agent layer with "
                               "long-term memory of the user translates personal signals so the fully frozen foundation "
                               "model can read them. Nemotron agents plan each day and improve the playbook over time.",
                     "kv": [["Who fixes drift", "The agent team"], ["Model changes", "Never"]]},
    "footnote": "Research prototype. In our MC_Maze test with simulated drift, the agent layer beat the frozen decoder on "
                "early drift days but not yet on all days; metric cards show design goals, not clinical results.",
}


def build(cfg, name):
    accent = {"existing": ("#A32D2D", "#FCEBEB", "rgba(226,75,74,.16)"),
              "agent": ("#0F6E56", "#E1F5EE", "rgba(29,158,117,.18)")}[name.split("_")[0]]
    html = (TPL.replace("__TITLE__", cfg["title"]).replace("__ACCENT__", accent[0])
            .replace("__ACCENT_BG__", accent[1]).replace("__ACCENT_BG_DARK__", accent[2])
            .replace("__CONFIG__", json.dumps(cfg, ensure_ascii=False)))
    path = os.path.join(HERE, f"{name}.html")
    open(path, "w").write(html)
    print("wrote", path)


build(existing, "existing_system")
build(agent, "agent_system")
