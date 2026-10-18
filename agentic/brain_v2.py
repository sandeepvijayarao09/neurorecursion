"""Memory layer v2: three label-free safeguards added to the memory-driven orchestrator.

  split_half     accept a proposal only if it improves the verifier on BOTH halves of
                 the verify slice (filters lucky single-slice wins)
  delayed_credit at the start of each session, re-score yesterday's accepted changes on
                 today's data; changes whose benefit did not persist cost their agent
                 trust in memory (Beta posterior), so the schedule learns from it
  gain_bounds    memory keeps the history of accepted input-gain sizes; a proposal far
                 outside this user's history is rejected

Each flag can be switched off to measure its effect. Nothing here touches the frozen model.
"""
import copy
import json

import numpy as np

from brain import ChannelMemoryAgent, Ctx, MemoryOrchestrator, state_from_dict, state_to_dict
from memory import UserMemory


class MemoryV2(UserMemory):
    def __init__(self, path, user):
        super().__init__(path, user)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (user TEXT, k TEXT, v TEXT, PRIMARY KEY (user, k))")

    def get(self, k, default=None):
        row = self.db.execute("SELECT v FROM kv WHERE user=? AND k=?", (self.user, k)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (self.user, k, json.dumps(v)))
        self.db.commit()


class MemoryOrchestratorV2(MemoryOrchestrator):
    def __init__(self, *a, split_half=True, delayed_credit=True, gain_bounds=True, warm_start=True,
                 warm_split=False, **kw):
        super().__init__(*a, **kw)
        self.split_half, self.delayed_credit, self.gain_bounds = split_half, delayed_credit, gain_bounds
        self.warm_start, self.warm_split = warm_start, warm_split

    def _accept(self, Xv, cur_state, new_state, cur):
        sc = self.verifier(Xv, new_state)
        ok = sc < cur * (1 - self.margin)
        if ok and self.split_half:
            h = len(Xv) // 2
            a0, b0 = self.verifier(Xv[:h], cur_state), self.verifier(Xv[h:], cur_state)
            a1, b1 = self.verifier(Xv[:h], new_state), self.verifier(Xv[h:], new_state)
            ok = a1 < a0 and b1 < b0
        return ok, sc

    def _gain_size(self, before, after):
        on = (before.ch_weight > 0) & (after.ch_weight > 0)
        return float(np.median(np.abs(np.log(after.ch_weight[on] / before.ch_weight[on])))) if on.any() else 0.0

    def run_day(self, day, X_adapt):
        n = len(X_adapt)
        Xf, Xv = X_adapt[: int(n * 0.6)], X_adapt[int(n * 0.6):]
        n_ch = X_adapt.shape[1]
        credit = []

        # delayed credit: did yesterday's accepted changes still help today?
        if self.delayed_credit:
            for p in self.mem.get("pending", []):
                b, a = state_from_dict(p["before"], self.template), state_from_dict(p["after"], self.template)
                held = self.verifier(Xv, a) < self.verifier(Xv, b)
                self.mem.update_belief(p["agent"], held)
                credit.append((p["agent"], bool(held)))
            self.mem.put("pending", [])

        from agents import ChannelHealthAgent
        ratio = Xf.mean(0) / np.maximum(self.ref_rate, 1e-6)
        dead, noisy = ChannelHealthAgent(self.ref_rate)._flag(Ctx(Xf), self.template)
        self.mem.update_channels(np.clip(ratio, 0, 10), set(dead) | set(noisy))

        fresh = copy.deepcopy(self.template)
        s_fresh = self.verifier(Xv, fresh)
        prev_day, prev = self.mem.last_good_state()
        start, s_start, source = fresh, s_fresh, "fresh"
        if prev is not None and self.warm_start:
            warm = state_from_dict(prev, self.template)
            s_warm = self.verifier(Xv, warm)
            ok_warm = s_warm < s_fresh
            if ok_warm and self.warm_split:          # remembered state must win on both halves too
                h = len(Xv) // 2
                ok_warm = (self.verifier(Xv[:h], warm) < self.verifier(Xv[:h], fresh) and
                           self.verifier(Xv[h:], warm) < self.verifier(Xv[h:], fresh))
            if ok_warm:
                start, s_start, source = warm, s_warm, f"memory ({prev_day})"

        order, pruned = self.mem.thompson_order(self.AGENTS, self.rng)
        import llm
        plan = llm.ask_json("planner", llm.PLANNER_MODEL,
                            "Order BCI maintenance agents for today. Use the user's memory. Return all candidates you want run.",
                            json.dumps({"candidates": order, "memory": self.mem.summary(n_ch),
                                        "reply_schema": {"steps": "list"}}), max_tokens=300)
        if plan and isinstance(plan.get("steps"), list):
            order = [a_ for a_ in plan["steps"] if a_ in order] or order
        state, cur, accepted, pending = start, s_start, [], []
        hist = self.mem.get("gain_hist", [])
        for name in order:
            a, mode = (name.split("/") + [None])[:2]
            if a == "channel_memory":
                new, info = self.agents[a].propose(Ctx(Xf), state)
            else:
                new, info = self.agents[a].propose(Ctx(Xf), state, self.master, mode=mode)
            ok, sc = self._accept(Xv, state, new, cur)
            why = "verifier"
            if ok and self.gain_bounds and name == "alignment/input":
                g = self._gain_size(state, new)
                bound = max(0.15, 2.0 * float(np.percentile(hist, 90))) if len(hist) >= 3 else 0.35
                if g > bound:
                    ok, why = False, f"gain {g:.2f} outside learned bound {bound:.2f}"
                else:
                    hist.append(g)
            dec = "accept" if ok else "reject"
            self.mem.record_proposal(day, name, {**info, "why": why}, cur, sc, dec)
            if ok:
                pending.append({"agent": name, "before": state_to_dict(state), "after": state_to_dict(new)})
                state, cur = new, sc
                accepted.append(name)
        self.mem.put("gain_hist", hist[-30:])
        self.mem.put("pending", pending)

        good = cur <= s_fresh
        self.mem.save_state(day, state_to_dict(state), cur, good)
        self.mem.record_episode(day, {"start": source, "order": order, "pruned": pruned, "credit": credit},
                                {"accepted": accepted, "score_fresh": s_fresh, "score_final": cur})
        self.mem.consolidate(day, n_ch)
        return state, {"start": source, "order": order, "pruned": pruned, "accepted": accepted, "credit": credit}
