"""Recursive agents for a self-maintaining brain-computer interface decoder.

Three loops:
  * Definition of done - a Nemotron planner turns the one-line goal ("keep the cursor
    usable today without recalibration") into runnable checks, tool agents work
    until the checks pass, and the planner decides continue / re-plan / stop / ping a human.
  * Version N+1        - after each version's runs a Nemotron reviewer reads the audit
    log and rewrites the playbook (thresholds, agent order, checks). The change is
    kept only if it beats the current version on held-out drift scenarios.
  * Shared memory      - every agent reads memory.json before it works and the reviewer
    updates it afterwards.

The decoder itself (pretrained spike encoder + linear head) is frozen after day 0.
Agents never see hand kinematics; ground truth only scores held-out scenarios.

Usage: python recursive.py <mc_maze.nwb> <out_dir> [rounds]
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import stage_b_spikes_to_cursor as sb  # noqa: E402
import llm  # noqa: E402
from agents import (AdaptState, AlignmentAgent, ChannelHealthAgent, DayContext,  # noqa: E402
                    FrozenMaster, IntentAgent, Verifier)

GOAL = "Keep the user's cursor usable today without asking them to recalibrate."
DAY0_TRIALS, TRIALS_PER_DAY, DAYS = 150, 25, 3
TRAIN_SCEN, HELDOUT_SCEN = [0, 1], [100, 101]
AGENTS = ["channel_health", "alignment/input", "alignment/latent", "intent"]

PLAYBOOK_V0 = {
    "version": 0,
    "dead_ratio": 0.2, "noisy_ratio": 3.0, "max_mask_frac": 1.0,
    "verifier_margin": 0.01, "coral_shrink": 1.0, "max_loops": 3,
    "agent_order": AGENTS[:],
    "disabled": [],
    "dod": {"min_intent_cos_gain": 0.0, "speed_range": [0.5, 2.0], "max_masked_frac": 0.5},
}


# ----------------------------------------------------------------------------- setup
class World:
    """Day-0 master model plus generators for drifted days."""

    def __init__(self, nwb, out_dir):
        units, ht, hv, hp, tr = sb.load_nwb(nwb)
        self.tbin, self.X, self.V, self.P = sb.bin_data(units, ht, hv, hp)
        self.tr = tr
        self.Vl = np.roll(self.V, -sb.LAG_BINS, 0)
        st, sp = tr["start_time"], tr["stop_time"]
        tp, act = tr["target_pos_list"], tr["active_target"].astype(int)
        self.tgt = np.array([tp[i][act[i]] for i in range(len(st))])
        self.cands = np.unique(np.round(self.tgt, -1), axis=0)
        inT = np.zeros(len(self.tbin), bool)
        for s, e in zip(st, sp):
            inT[(self.tbin >= s) & (self.tbin < e)] = True
        d0 = self.tbin < st[DAY0_TRIALS]
        ck = os.path.join(out_dir, "encoder_day0.pt")
        if os.path.exists(ck):
            self.enc = sb.SpikeEncoder(self.X.shape[1]).to(sb.DEV)
            self.enc.load_state_dict(torch.load(ck, map_location=sb.DEV))
        else:
            print("Day 0: pretraining encoder (one time)…", flush=True)
            self.enc, _, _ = sb.pretrain(self.X[d0], self.X.shape[1], epochs=12)
            torch.save(self.enc.state_dict(), ck)
        embed = lambda A: sb.embed_all(self.enc, A.astype(np.float32))  # noqa: E731
        E0 = embed(self.X[d0])
        cut = np.flatnonzero(d0 & inT)
        k = int(len(cut) * 0.85)
        Zh = sb.history(E0)
        head = sb.best_ridge(Zh[cut[:k]], self.Vl[cut[:k]], Zh[cut[k:]], self.Vl[cut[k:]])
        self.master = FrozenMaster(embed, head, sb.history, sb.LAG_BINS)
        self.ref_rate = self.X[d0].mean(0)
        self.ref_mu, self.ref_cov = E0.mean(0), np.cov(E0.T) + 1e-3 * np.eye(E0.shape[1])
        self.d_lat = E0.shape[1]
        s0 = AdaptState(self.X.shape[1], self.d_lat)
        V0 = self.master.decode(self.X[d0], s0)
        w0 = self._windows(range(DAY0_TRIALS), self.tbin[d0])
        self.ref_speed = float(np.median(np.linalg.norm(np.concatenate([V0[a:b] for a, b in w0]), axis=1)))
        self._days = {}

    def _windows(self, idx, t):
        go = self.tr["go_cue_time"]
        return [(int(np.searchsorted(t, go[i] + 0.25)), int(np.searchsorted(t, go[i] + 1.25))) for i in idx]

    def day(self, scen, d):
        """Scenario `scen` gives a different random drift pattern; day d sets severity."""
        key = (scen, d)
        if key in self._days:
            return self._days[key]
        rng = np.random.default_rng(1000 + scen)
        perm = rng.permutation(self.X.shape[1])
        gains = rng.normal(0, 0.2, self.X.shape[1])
        rng_d = np.random.default_rng(scen * 10 + d)
        st, sp = self.tr["start_time"], self.tr["stop_time"]
        ids = np.arange(DAY0_TRIALS + (d - 1) * TRIALS_PER_DAY, DAY0_TRIALS + d * TRIALS_PER_DAY)
        m = (self.tbin >= st[ids[0]]) & (self.tbin < sp[ids[-1]])
        Xd = rng_d.poisson(self.X[m] * np.exp(gains * d)[None, :]).astype(np.float32)
        Xd[:, perm[: 6 * d]] = 0
        noisy = perm[60: 60 + 3 * d]
        Xd[:, noisy] += rng_d.poisson(0.4, (len(Xd), len(noisy)))
        t_day, P_day = self.tbin[m], self.P[m]
        win = self._windows(ids, t_day)
        start = np.array([P_day[min(a, len(P_day) - 1)] for a, _ in win])
        dv = self.tgt[ids] - start
        local = np.arange(len(ids))
        ctx = DayContext(Xd, t_day, win, dv / np.linalg.norm(dv, axis=1, keepdims=True), local[:10], local[10:17])
        ev = np.zeros(len(t_day), bool)
        for i in local[17:]:
            ev[(t_day >= st[ids[i]]) & (t_day < sp[ids[i]])] = True
        self._days[key] = (ctx, {"V": self.V[m], "ev": ev})
        return self._days[key]

    def r2(self, scen, d, state):
        ctx, truth = self.day(scen, d)
        Vd = self.master.decode(ctx.X, state)
        return sb.r2(truth["V"][truth["ev"]], Vd[truth["ev"]])


# ----------------------------------------------------------------------------- shared brain
class SharedBrain:
    def __init__(self, path):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {"notes": [], "agent_stats": {}, "reverted_edits": []}

    def read(self):
        return {"notes": self.data["notes"][-12:], "agent_stats": self.data["agent_stats"],
                "reverted_edits": self.data.get("reverted_edits", [])[-6:]}

    def update(self, notes=(), stats=None):
        for n in notes:
            if n and n not in self.data["notes"]:
                self.data["notes"].append(n)
        self.data["notes"] = self.data["notes"][-20:]      # stay lean
        if stats:
            self.data["agent_stats"] = stats
        json.dump(self.data, open(self.path, "w"), indent=2)


# ----------------------------------------------------------------------------- planner (Done Means Done)
PLANNER_SYS = (
    "You plan the daily maintenance of a frozen brain-computer-interface cursor decoder. "
    "Tool agents: channel_health (mask dead/noisy channels), alignment/input (per-channel gain), "
    "alignment/latent (CORAL feature alignment), intent (rotate output toward task targets). "
    "You never see ground truth. Turn the goal into checks the verifier can run, then pick agent order.")


def plan_day(goal, playbook, brain, telemetry):
    user = json.dumps({"goal": goal, "playbook": playbook, "shared_memory": brain.read(), "today": telemetry,
                       "reply_schema": {"definition_of_done": {"min_intent_cos_gain": "float",
                                                               "speed_range": "[lo, hi]",
                                                               "max_masked_frac": "float"},
                                        "steps": "list of agent names in order (subset of agent_order)",
                                        "reason": "one sentence"}})
    out = llm.ask_json("planner", llm.PLANNER_MODEL, PLANNER_SYS, user)
    if out and isinstance(out.get("steps"), list):
        steps = [s for s in out["steps"] if s in AGENTS and s not in playbook["disabled"]]
        dod = {**playbook["dod"], **(out.get("definition_of_done") or {})}
        return {"steps": steps or playbook["agent_order"], "dod": dod, "reason": out.get("reason", ""), "by": "nemotron"}
    steps = [s for s in playbook["agent_order"] if s not in playbook["disabled"]]
    if telemetry["flagged_channels"] == 0 and "channel_health" in steps:
        steps.remove("channel_health")
    return {"steps": steps, "dod": playbook["dod"], "reason": "rule-based plan (no CRUSOE_API_KEY)", "by": "rules"}


def decide_next(goal, dod_report, history):
    """Called when a loop ends without meeting the definition of done."""
    out = llm.ask_json("planner", llm.PLANNER_MODEL, PLANNER_SYS,
                       json.dumps({"goal": goal, "dod_report": dod_report, "recent": history[-6:],
                                   "reply_schema": {"action": "continue|replan|stop|escalate", "reason": "str"}}),
                       max_tokens=400)
    if out and out.get("action") in {"continue", "replan", "stop", "escalate"}:
        return out["action"], out.get("reason", ""), "nemotron"
    if not dod_report["checks"]["masked_ok"] or not dod_report["checks"]["speed_ok"]:
        return "escalate", "safety check failing; a human should look", "rules"
    return "stop", "no further accepted proposals", "rules"


def check_dod(dod, start, cur, state):
    masked_frac = float((state.ch_weight == 0).mean())
    checks = {
        "gain_ok": cur["intent_cos"] - start["intent_cos"] >= dod["min_intent_cos_gain"],
        "speed_ok": dod["speed_range"][0] <= cur["speed_ratio"] <= dod["speed_range"][1],
        "masked_ok": masked_frac <= dod["max_masked_frac"],
    }
    return {"done": all(checks.values()), "checks": checks, "masked_frac": masked_frac,
            "intent_gain": cur["intent_cos"] - start["intent_cos"]}


# ----------------------------------------------------------------------------- one run (a day)
def run_day(world, playbook, brain, scen, d, state, log, use_llm=True):
    ctx, _ = world.day(scen, d)
    agents = {"channel_health": ChannelHealthAgent(world.ref_rate, playbook["dead_ratio"],
                                                   playbook["noisy_ratio"], playbook["max_mask_frac"]),
              "alignment": AlignmentAgent(world.ref_rate, world.ref_mu, world.ref_cov, playbook["coral_shrink"]),
              "intent": IntentAgent(world.ref_speed)}
    ver = Verifier(world.ref_speed, playbook["verifier_margin"])
    start = cur = ver.score(ctx, state, world.master)
    dead, noisy = agents["channel_health"]._flag(ctx, state)
    telemetry = {"day": d, "flagged_channels": int(len(dead) + len(noisy)),
                 "masked_now": int((state.ch_weight == 0).sum()), "verifier": start}
    plan = plan_day(GOAL, playbook, brain, telemetry) if use_llm else plan_day.__wrapped__(playbook, telemetry)
    log({"scen": scen, "day": d, "event": "plan", **plan})
    tried, hist = set(), []
    action = "stop"
    for loop in range(playbook["max_loops"]):
        accepted = False
        for step in plan["steps"]:
            if step in tried:
                continue
            name, mode = (step.split("/") + [None])[:2]
            for attempt in range(2):
                if name == "channel_health":
                    new, info = agents[name].propose(ctx, state, attempt=attempt)
                elif name == "alignment":
                    new, info = agents[name].propose(ctx, state, world.master, attempt=attempt, mode=mode)
                else:
                    new, info = agents[name].propose(ctx, state, world.master, attempt=attempt)
                sc = ver.score(ctx, new, world.master)
                dec, why = ver.decide(cur, sc)
                if dec == "accept" and (new.ch_weight == 0).mean() > plan["dod"]["max_masked_frac"]:
                    dec, why = "reject", "would exceed masking limit in definition of done"
                rec = {"scen": scen, "day": d, "loop": loop, "agent": step, "attempt": attempt,
                       "decision": dec, "why": why, "delta": sc["intent_cos"] - cur["intent_cos"]}
                log(rec)
                hist.append(rec)
                if dec == "accept":
                    state, cur, accepted = new, sc, True
                    tried.add(step)
                    break
                if dec == "reject":
                    tried.add(step)
                    break
            else:
                tried.add(step)
        rep = check_dod(plan["dod"], start, cur, state)
        if rep["done"] and accepted is False:
            action = "done"
            break
        if not accepted:
            action, why, by = decide_next(GOAL, rep, hist)
            log({"scen": scen, "day": d, "event": "decide", "action": action, "why": why, "by": by, "dod": rep})
            if action in {"stop", "escalate"}:
                break
            if action == "replan":
                plan = plan_day(GOAL, playbook, brain, {**telemetry, "verifier": cur, "replan": True})
                tried = set()
    rep = check_dod(plan["dod"], start, cur, state)
    log({"scen": scen, "day": d, "event": "end", "action": action, "dod": rep})
    return state, rep, action


plan_day.__wrapped__ = lambda playbook, telemetry: {
    "steps": [s for s in playbook["agent_order"] if s not in playbook["disabled"]],
    "dod": playbook["dod"], "reason": "fixed playbook", "by": "rules"}


def run_version(world, playbook, brain, scens, log, use_llm=True):
    """Run all days for each scenario, carrying state day to day. Returns R² and stats."""
    r2s, frozen, records, escal = [], [], [], 0
    for scen in scens:
        state = AdaptState(world.X.shape[1], world.d_lat)
        for d in range(1, DAYS + 1):
            local = []
            state, rep, action = run_day(world, playbook, brain, scen, d, state,
                                         lambda r: (log(r), local.append(r)), use_llm)
            escal += action == "escalate"
            records += local
            r2s.append(world.r2(scen, d, state))
            frozen.append(world.r2(scen, d, AdaptState(world.X.shape[1], world.d_lat)))
    return {"r2": float(np.mean(r2s)), "frozen_r2": float(np.mean(frozen)), "per_day": r2s,
            "escalations": escal, "records": records}


# ----------------------------------------------------------------------------- reviewer (Version N+1)
REVIEWER_SYS = (
    "You improve a playbook for agents that maintain a frozen BCI decoder. You see per-agent "
    "accept/reject statistics and label-free verifier deltas from the last runs. Propose small edits "
    "to playbook fields only. Allowed keys: dead_ratio, noisy_ratio, max_mask_frac, verifier_margin, "
    "coral_shrink, agent_order, disabled, dod. Also give up to 3 short memory notes for other agents.")


def agent_stats(records):
    st = {}
    for r in records:
        if "decision" not in r:
            continue
        s = st.setdefault(r["agent"], {"accept": 0, "reject": 0, "iterate": 0, "mean_delta": []})
        s[r["decision"]] += 1
        s["mean_delta"].append(r["delta"])
    for s in st.values():
        s["mean_delta"] = round(float(np.mean(s["mean_delta"])), 4)
    return st


def review(playbook, stats, train_result, brain):
    tried = brain.data.get("reverted_edits", [])
    user = json.dumps({"playbook": playbook, "agent_stats": stats, "shared_memory": brain.read(),
                       "do_not_repeat": tried,
                       "label_free_summary": {"escalations": train_result["escalations"]},
                       "reply_format_example": {"edits": {"verifier_margin": 0.02, "disabled": ["intent"]},
                                                "rationale": "one sentence", "memory_notes": ["short fact"]}})
    out = llm.ask_json("reviewer", llm.REVIEWER_MODEL,
                       REVIEWER_SYS + " Change at most 2 fields. Return only the changed fields under 'edits'.", user)
    if out and not isinstance(out.get("edits"), dict) and isinstance(out.get("playbook"), dict):
        out["edits"] = {k: v for k, v in out["playbook"].items() if playbook.get(k) != v}   # full playbook -> diff
    if out and isinstance(out.get("edits"), dict) and out["edits"]:
        edits = {k: v for k, v in out["edits"].items() if k in PLAYBOOK_V0 and k != "version"}
        return edits, out.get("rationale", ""), out.get("memory_notes", [])[:3], "nemotron"
    # rule-based reviewer: drop agents that mostly hurt, cap aggressive masking
    edits, notes = {}, []
    for a, s in stats.items():
        n = s["accept"] + s["reject"] + s["iterate"]
        if n >= 3 and s["accept"] / n < 0.2 and s["mean_delta"] < 0 and a not in playbook["disabled"]:
            edits.setdefault("disabled", list(playbook["disabled"])).append(a)
            notes.append(f"{a} rarely helps (accept {s['accept']}/{n}, mean delta {s['mean_delta']}); skip it")
    ch = stats.get("channel_health")
    if ch and ch["reject"] > ch["accept"]:
        edits["max_mask_frac"] = round(max(0.05, playbook["max_mask_frac"] * 0.5), 3)
        notes.append("channel_health over-masks; cap channels masked per proposal")
    fallbacks = [({"verifier_margin": round(playbook["verifier_margin"] * 2, 4)}, "raise verifier margin to resist noisy accepts"),
                 ({"coral_shrink": 0.5}, "soften latent alignment"),
                 ({"max_mask_frac": 0.15}, "cap masking hard"),
                 ({"disabled": sorted(set(playbook["disabled"]) | {"intent"})}, "skip intent rotation")]
    if not edits or edits in tried:
        for e, n in fallbacks:
            if e not in tried:
                edits, notes = e, [n]
                break
    return edits, "rule-based review (no CRUSOE_API_KEY)", notes, "rules"


# ----------------------------------------------------------------------------- main loop
def main(nwb, out_dir, rounds=2):
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "recursive_audit.jsonl"), "w")

    def log(r):
        r["ts"] = time.strftime("%H:%M:%S")
        logf.write(json.dumps(r, default=float) + "\n")
        logf.flush()

    t0 = time.time()
    print(f"LLM backend: {'Crusoe ' + llm.PLANNER_MODEL if llm.available() else 'rule-based fallback (set CRUSOE_API_KEY)'}")
    world = World(nwb, out_dir)
    brain = SharedBrain(os.path.join(out_dir, "memory.json"))
    current = copy.deepcopy(PLAYBOOK_V0)
    held = run_version(world, current, brain, HELDOUT_SCEN, log)
    board = [{"version": 0, "heldout_r2": held["r2"], "frozen_r2": held["frozen_r2"], "kept": True,
              "edits": {}, "by": "-"}]
    print(f"v0 held-out R² {held['r2']:.3f} (frozen decoder {held['frozen_r2']:.3f})", flush=True)

    for rnd in range(1, rounds + 1):
        train = run_version(world, current, brain, TRAIN_SCEN, log)
        stats = agent_stats(train["records"])
        edits, why, notes, by = review(current, stats, train, brain)
        cand = copy.deepcopy(current)
        cand.update(edits)
        cand["version"] = current["version"] + 1
        cheld = run_version(world, cand, brain, HELDOUT_SCEN, log)
        keep = cheld["r2"] > held["r2"] + 1e-3
        log({"event": "review", "round": rnd, "edits": edits, "rationale": why, "by": by,
             "candidate_heldout_r2": cheld["r2"], "current_heldout_r2": held["r2"], "kept": keep})
        if not keep:
            brain.data.setdefault("reverted_edits", []).append(edits)
        brain.update(notes + ([f"v{cand['version']} kept: {edits}"] if keep else
                              [f"v{cand['version']} reverted, edits {edits} did not help on held-out"]), stats)
        board.append({"version": cand["version"], "heldout_r2": cheld["r2"], "frozen_r2": cheld["frozen_r2"],
                      "kept": keep, "edits": edits, "by": by, "rationale": why})
        print(f"round {rnd}: {by} reviewer edits {edits} -> held-out R² {cheld['r2']:.3f} "
              f"vs {held['r2']:.3f} -> {'KEEP' if keep else 'revert'}", flush=True)
        if keep:
            current, held = cand, cheld

    json.dump({"scoreboard": board, "final_playbook": current, "llm_calls": llm.CALLS,
               "runtime_s": round(time.time() - t0, 1)},
              open(os.path.join(out_dir, "scoreboard.json"), "w"), indent=2, default=float)
    json.dump(current, open(os.path.join(out_dir, "playbook_final.json"), "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s; final playbook v{current['version']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
