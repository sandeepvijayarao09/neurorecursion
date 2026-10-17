"""Per-user memory layer for the agent team (SQLite, stdlib only).

Four kinds of memory, all keyed by user:

  episodic    episodes, proposals        what happened each day, every proposal + verdict
  semantic    channel_stats, agent_beliefs  channel reliability history (EWMA), per-agent
                                           Beta posterior of "this agent helps this user"
  procedural  states                     last adaptation state that passed verification
  notes       notes                      short facts for the LLM planner, deduplicated

Agents read `UserMemory` before they act and the orchestrator calls `consolidate()`
after each day. Nothing here touches the frozen model.
"""
import json
import sqlite3
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes  (user TEXT, day TEXT, ts REAL, telemetry TEXT, outcome TEXT);
CREATE TABLE IF NOT EXISTS proposals (user TEXT, day TEXT, agent TEXT, info TEXT,
                                      before REAL, after REAL, decision TEXT);
CREATE TABLE IF NOT EXISTS channel_stats (user TEXT, ch INTEGER, ewma_ratio REAL, flags INTEGER,
                                          seen INTEGER, PRIMARY KEY (user, ch));
CREATE TABLE IF NOT EXISTS agent_beliefs (user TEXT, agent TEXT, alpha REAL, beta REAL,
                                          trials INTEGER, PRIMARY KEY (user, agent));
CREATE TABLE IF NOT EXISTS states (user TEXT, day TEXT, state TEXT, score REAL, good INTEGER);
CREATE TABLE IF NOT EXISTS notes (user TEXT, ts REAL, topic TEXT, text TEXT, source TEXT, PRIMARY KEY (user, topic));
"""


class UserMemory:
    def __init__(self, path, user, ewma=0.4):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.user, self.ewma = user, ewma

    # ------------------------------------------------------------------ episodic
    def record_episode(self, day, telemetry, outcome):
        self.db.execute("INSERT INTO episodes VALUES (?,?,?,?,?)",
                        (self.user, day, time.time(), json.dumps(telemetry, default=float),
                         json.dumps(outcome, default=float)))
        self.db.commit()

    def record_proposal(self, day, agent, info, before, after, decision):
        self.db.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?,?)",
                        (self.user, day, agent, json.dumps(info, default=float), before, after, decision))

    def episodes(self, n=10):
        rows = self.db.execute("SELECT day, telemetry, outcome FROM episodes WHERE user=? "
                               "ORDER BY ts DESC LIMIT ?", (self.user, n)).fetchall()
        return [(d, json.loads(t), json.loads(o)) for d, t, o in rows]

    # ------------------------------------------------------------------ semantic: channels
    def update_channels(self, rate_ratio, flagged):
        for ch, r in enumerate(rate_ratio):
            row = self.db.execute("SELECT ewma_ratio, flags, seen FROM channel_stats WHERE user=? AND ch=?",
                                  (self.user, ch)).fetchone()
            f = int(ch in flagged)
            if row is None:
                self.db.execute("INSERT INTO channel_stats VALUES (?,?,?,?,?)", (self.user, ch, float(r), f, 1))
            else:
                e = (1 - self.ewma) * row[0] + self.ewma * float(r)
                self.db.execute("UPDATE channel_stats SET ewma_ratio=?, flags=?, seen=? WHERE user=? AND ch=?",
                                (e, row[1] + f, row[2] + 1, self.user, ch))
        self.db.commit()

    def channel_reliability(self, n_ch):
        """1.0 = never flagged; low values = repeatedly flagged across days."""
        rel = np.ones(n_ch)
        for ch, flags, seen in self.db.execute("SELECT ch, flags, seen FROM channel_stats WHERE user=?",
                                               (self.user,)):
            rel[ch] = 1 - (flags + 0.0) / (seen + 1.0)        # +1 prior keeps single flags from masking
        return rel

    def channel_ewma(self, n_ch):
        e = np.ones(n_ch)
        for ch, v in self.db.execute("SELECT ch, ewma_ratio FROM channel_stats WHERE user=?", (self.user,)):
            e[ch] = v
        return e

    # ------------------------------------------------------------------ semantic: agent efficacy
    def belief(self, agent):
        row = self.db.execute("SELECT alpha, beta, trials FROM agent_beliefs WHERE user=? AND agent=?",
                              (self.user, agent)).fetchone()
        return row if row else (1.0, 1.0, 0)

    def update_belief(self, agent, success):
        a, b, n = self.belief(agent)
        a, b = a + float(success), b + float(not success)
        self.db.execute("INSERT OR REPLACE INTO agent_beliefs VALUES (?,?,?,?,?)", (self.user, agent, a, b, n + 1))

    def thompson_order(self, agents, rng, prune_below=0.2, min_trials=3):
        """Data-driven agent schedule: sample each agent's success rate, run best first,
        skip agents whose posterior mean is very low after enough evidence."""
        draws, skipped = [], []
        for ag in agents:
            a, b, n = self.belief(ag)
            if n >= min_trials and a / (a + b) < prune_below:
                skipped.append(ag)
                continue
            draws.append((rng.beta(a, b), ag))
        return [ag for _, ag in sorted(draws, reverse=True)], skipped

    # ------------------------------------------------------------------ procedural: states
    def save_state(self, day, state_dict, score, good):
        self.db.execute("INSERT INTO states VALUES (?,?,?,?,?)",
                        (self.user, day, json.dumps(state_dict), score, int(good)))
        self.db.commit()

    def last_good_state(self):
        row = self.db.execute("SELECT day, state FROM states WHERE user=? AND good=1 ORDER BY rowid DESC LIMIT 1",
                              (self.user,)).fetchone()
        return (row[0], json.loads(row[1])) if row else (None, None)

    # ------------------------------------------------------------------ notes
    def note(self, topic, text, source="consolidate"):
        """One note per topic: newer evidence overwrites older, so memory stays lean."""
        self.db.execute("INSERT OR REPLACE INTO notes VALUES (?,?,?,?,?)", (self.user, time.time(), topic, text, source))

    def notes(self, n=8):
        return [t for (t,) in self.db.execute("SELECT text FROM notes WHERE user=? ORDER BY ts DESC LIMIT ?",
                                              (self.user, n))]

    # ------------------------------------------------------------------ consolidation
    def consolidate(self, day, n_ch):
        """After a day: update agent beliefs from today's proposals, derive notes."""
        rows = self.db.execute("SELECT agent, before, after, decision FROM proposals WHERE user=? AND day=?",
                               (self.user, day)).fetchall()
        for agent, before, after, dec in rows:
            self.update_belief(agent, dec == "accept")
        rel = self.channel_reliability(n_ch)
        bad = np.flatnonzero(rel < 0.5)
        if len(bad):
            self.note("bad_channels", f"{len(bad)} channels chronically unreliable: {bad[:12].tolist()}")
        for agent, a, b, n in self.db.execute("SELECT agent, alpha, beta, trials FROM agent_beliefs WHERE user=?",
                                              (self.user,)).fetchall():
            if n >= 3:
                self.note(f"agent:{agent}", f"{agent}: helps this user {a / (a + b):.0%} of the time over {n} tries")
        self.db.commit()

    def summary(self, n_ch):
        """Compact context for an LLM planner (keeps prompts lean)."""
        rel = self.channel_reliability(n_ch)
        return {"days_seen": len(self.episodes(1000)),
                "chronically_bad_channels": int((rel < 0.5).sum()),
                "agent_beliefs": {a: round(al / (al + be), 2) for a, al, be in
                                  self.db.execute("SELECT agent, alpha, beta FROM agent_beliefs WHERE user=?",
                                                  (self.user,))},
                "notes": self.notes()}
