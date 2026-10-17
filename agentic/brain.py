"""Memory-driven orchestrator: the agent team that improves with every day it sees a user.

Per day:
  1. sense       - channel telemetry -> written to memory (channel EWMA + flags)
  2. recall      - candidates: fresh state vs last verified state from memory;
                   verifier picks the better start (warm start, or rollback when stale)
  3. schedule    - agent order by Thompson sampling on this user's Beta posteriors;
                   agents with low posterior after enough tries are pruned
  4. act/verify  - each agent proposes once; label-free verifier accepts only if better
  5. consolidate - proposals update beliefs; verified state saved; notes for the planner

An optional LLM planner (Nemotron on Crusoe via llm.py) can reorder agents using
memory.summary(); without CRUSOE_API_KEY the data-driven schedule is used as is.
"""
import copy
import json

import numpy as np

import llm
from agents import AdaptState, AlignmentAgent, ChannelHealthAgent


class Ctx:
    def __init__(self, X):
        self.X = X


def state_to_dict(s):
    return {"ch_weight": s.ch_weight.tolist(), "lat_A": s.lat_A.tolist(), "lat_mu": s.lat_mu.tolist(),
            "lat_mu0": s.lat_mu0.tolist(), "out_M": s.out_M.tolist()}


def state_from_dict(d, template):
    s = copy.deepcopy(template)
    s.ch_weight = np.array(d["ch_weight"], np.float32)
    s.lat_A, s.lat_mu = np.array(d["lat_A"], np.float32), np.array(d["lat_mu"], np.float32)
    s.lat_mu0, s.out_M = np.array(d["lat_mu0"], np.float32), np.array(d["out_M"], np.float32)
    return s


class ChannelMemoryAgent:
    """Channel health that uses history: masks channels flagged today OR chronically
    unreliable in memory, and restores masked channels that have recovered."""
    name = "channel_memory"

    def __init__(self, ref_rate, mem, max_mask_frac=0.25):
        self.ref, self.mem, self.cap = ref_rate, mem, max_mask_frac
        self.today = ChannelHealthAgent(ref_rate, 0.2, 3.0, max_mask_frac)

    def propose(self, ctx, state):
        n = len(self.ref)
        dead, noisy = self.today._flag(ctx, AdaptState(n, state.d_lat, n_out=state.n_out))
        rel = self.mem.channel_reliability(n)
        bad = set(dead) | set(noisy) | set(np.flatnonzero(rel < 0.5))
        rate = ctx.X.mean(0)
        ratio = rate / np.maximum(self.ref, 1e-6)
        recovered = [c for c in np.flatnonzero(state.ch_weight == 0)
                     if c not in bad and 0.5 < ratio[c] < 2.0]
        bad = sorted(bad)[: int(self.cap * n)]
        new = copy.deepcopy(state)
        # weights = per-channel gain x (n / kept) mask rescale; recover the gains first
        old_kept = max(1, (state.ch_weight > 0).sum())
        g = np.where(state.ch_weight > 0, state.ch_weight * old_kept / n, 1.0)
        g[bad] = 0
        kept = max(1, (g > 0).sum())
        new.ch_weight = (g * n / kept).astype(np.float32)
        return new, {"masked": len(bad), "from_memory": int((rel < 0.5).sum()), "restored": len(recovered)}


class MemoryOrchestrator:
    AGENTS = ["channel_memory", "alignment/input", "alignment/latent"]

    def __init__(self, master, verifier, mem, ref_rate, ref_mu, ref_cov, template, margin=0.02, seed=0):
        self.master, self.verifier, self.mem = master, verifier, mem
        self.template, self.margin = template, margin
        self.rng = np.random.default_rng(seed)
        self.agents = {"channel_memory": ChannelMemoryAgent(ref_rate, mem),
                       "alignment": AlignmentAgent(ref_rate, ref_mu, ref_cov)}
        self.ref_rate = ref_rate

    def run_day(self, day, X_adapt):
        n = len(X_adapt)
        Xf, Xv = X_adapt[: int(n * 0.6)], X_adapt[int(n * 0.6):]
        n_ch = X_adapt.shape[1]

        # 1. sense
        ratio = Xf.mean(0) / np.maximum(self.ref_rate, 1e-6)
        dead, noisy = ChannelHealthAgent(self.ref_rate)._flag(Ctx(Xf), self.template)
        self.mem.update_channels(np.clip(ratio, 0, 10), set(dead) | set(noisy))

        # 2. recall: warm start vs fresh, decided by the verifier
        fresh = copy.deepcopy(self.template)
        s_fresh = self.verifier(Xv, fresh)
        prev_day, prev = self.mem.last_good_state()
        start, s_start, source = fresh, s_fresh, "fresh"
        if prev is not None:
            warm = state_from_dict(prev, self.template)
            s_warm = self.verifier(Xv, warm)
            if s_warm < s_fresh:
                start, s_start, source = warm, s_warm, f"memory ({prev_day})"

        # 3. schedule from this user's evidence (optionally re-ranked by the LLM planner)
        order, pruned = self.mem.thompson_order(self.AGENTS, self.rng)
        plan = llm.ask_json("planner", llm.PLANNER_MODEL,
                            "Order BCI maintenance agents for today. Use the user's memory.",
                            json.dumps({"candidates": order, "memory": self.mem.summary(n_ch),
                                        "reply_schema": {"steps": "list"}}), max_tokens=300)
        if plan and isinstance(plan.get("steps"), list):
            order = [a for a in plan["steps"] if a in order] or order

        # 4. act + verify
        state, cur, accepted = start, s_start, []
        for name in order:
            a, mode = (name.split("/") + [None])[:2]
            if a == "channel_memory":
                new, info = self.agents[a].propose(Ctx(Xf), state)
            else:
                new, info = self.agents[a].propose(Ctx(Xf), state, self.master, mode=mode)
            sc = self.verifier(Xv, new)
            dec = "accept" if sc < cur * (1 - self.margin) else "reject"
            self.mem.record_proposal(day, name, info, cur, sc, dec)
            if dec == "accept":
                state, cur = new, sc
                accepted.append(name)

        # 5. consolidate
        good = cur <= s_fresh
        self.mem.save_state(day, state_to_dict(state), cur, good)
        self.mem.record_episode(day, {"flagged": len(dead) + len(noisy), "start": source, "order": order,
                                      "pruned": pruned},
                                {"accepted": accepted, "score_fresh": s_fresh, "score_final": cur})
        self.mem.consolidate(day, n_ch)
        return state, {"start": source, "order": order, "pruned": pruned, "accepted": accepted,
                       "score_fresh": s_fresh, "score_final": cur}
