"""Agentic adaptation layer between clean spikes and a frozen decoder.

The master model (pretrained spike encoder + linear velocity head) is trained
once on day 0 and never modified. Each later day, agents propose small changes
to an *adaptation state* that sits around the frozen model:

    spikes --(channel weights)--> encoder --(latent transform)--> head --(output map)--> cursor

  ChannelHealthAgent  -> mask dead / noisy channels, reweight the rest
  AlignmentAgent      -> per-channel gain matching (input) or CORAL latent alignment
  IntentAgent         -> pseudo-labels from the task (direction to the known target)
                         -> rotation + gain on the decoded velocity
  Verifier            -> accepts a proposal only if label-free scores improve on a
                         held-out slice of the day's own unlabeled data
  Orchestrator        -> picks agents, reads verifier scores, recurses <= N loops,
                         writes an audit log of every state change

No agent sees hand kinematics. Ground truth is used only by the evaluation code.
"""
import copy
import json
import time
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- state
@dataclass
class AdaptState:
    n_ch: int
    d_lat: int
    ch_weight: np.ndarray = None          # per-channel input multiplier (0 = masked)
    lat_A: np.ndarray = None              # latent affine: z' = (z - mu) @ A + mu0
    lat_mu: np.ndarray = None
    lat_mu0: np.ndarray = None
    out_M: np.ndarray = None              # map on decoded velocity (n_out x n_out)
    history: list = field(default_factory=list)
    n_out: int = 2

    def __post_init__(self):
        if self.ch_weight is None:
            self.ch_weight = np.ones(self.n_ch, np.float32)
        if self.lat_A is None:
            self.lat_A = np.eye(self.d_lat, dtype=np.float32)
            self.lat_mu = np.zeros(self.d_lat, np.float32)
            self.lat_mu0 = np.zeros(self.d_lat, np.float32)
        if self.out_M is None:
            self.out_M = np.eye(self.n_out, dtype=np.float32)

    def summary(self):
        return {"masked_channels": int((self.ch_weight == 0).sum()),
                "gain_range": [float(self.ch_weight[self.ch_weight > 0].min()),
                               float(self.ch_weight[self.ch_weight > 0].max())],
                "latent_transform": not np.allclose(self.lat_A, np.eye(self.d_lat)),
                "out_rotation_deg": float(np.degrees(np.arctan2(self.out_M[0, 1], self.out_M[0, 0]))),
                "out_gain": float(np.sqrt(abs(np.linalg.det(self.out_M))))}


class FrozenMaster:
    """Wraps the day-0 encoder + head. decode() applies an AdaptState around it."""

    def __init__(self, embed_fn, head_fn, history_fn, lag):
        self.embed_fn, self.head_fn, self.history_fn, self.lag = embed_fn, head_fn, history_fn, lag
        self._cache = {}

    def latent(self, X, state):
        key = (id(X), state.ch_weight.tobytes())
        if key not in self._cache:
            self._cache = {key: self.embed_fn(X * state.ch_weight[None, :])}   # keep cache small
        return self._cache[key]

    def decode(self, X, state):
        Z = self.latent(X, state)
        Z = (Z - state.lat_mu) @ state.lat_A + state.lat_mu0
        V = self.head_fn(self.history_fn(Z))
        V = np.roll(V, self.lag, 0)
        return V @ state.out_M


# ----------------------------------------------------------------------------- task context
@dataclass
class DayContext:
    """What the BCI software knows on a new day without any calibration labels:
    the spikes, trial timing from the task, and where the targets were."""
    X: np.ndarray            # [T, C] binned spikes for the day
    tbin: np.ndarray
    windows: list            # (a, b) bin index of the expected movement window per trial
    intent_dir: np.ndarray   # [n_trials, 2] unit vector from start to known target
    fit_trials: np.ndarray   # trial indices agents may fit on
    verify_trials: np.ndarray  # trial indices the verifier scores on


def window_mask(ctx, trials):
    m = np.zeros(len(ctx.tbin), bool)
    for i in trials:
        a, b = ctx.windows[i]
        m[a:b] = True
    return m


def pseudo_labels(ctx, trials):
    """Intent proxy: during each trial's movement window the user intends to move
    toward the target. Returns bin mask and unit intent direction per bin."""
    idx, dirs = [], []
    for i in trials:
        a, b = ctx.windows[i]
        idx.append(np.arange(a, b))
        dirs.append(np.repeat(ctx.intent_dir[i][None], b - a, 0))
    return np.concatenate(idx), np.concatenate(dirs)


# ----------------------------------------------------------------------------- agents
class ChannelHealthAgent:
    name = "channel_health"

    def __init__(self, ref_rate, dead_ratio=0.2, noisy_ratio=3.0, max_mask_frac=1.0):
        self.ref = ref_rate                 # day-0 mean count per bin per channel
        self.dead_ratio, self.noisy_ratio = dead_ratio, noisy_ratio
        self.max_mask_frac = max_mask_frac  # cap on channels masked per proposal

    def needed(self, ctx, state):
        return len(self._flag(ctx, state)[0]) + len(self._flag(ctx, state)[1]) > 0

    def _flag(self, ctx, state):
        rate = ctx.X.mean(0)
        live = self.ref > 0.01
        ratio = np.where(live, rate / np.maximum(self.ref, 1e-6), 1.0)
        dead = np.flatnonzero(live & (ratio < self.dead_ratio) & (state.ch_weight > 0))
        noisy = np.flatnonzero((ratio > self.noisy_ratio) & (state.ch_weight > 0))
        return dead, noisy

    def propose(self, ctx, state, attempt=0):
        if attempt:                          # iterate: loosen thresholds
            self.dead_ratio, self.noisy_ratio = 0.35, 2.0
        dead, noisy = self._flag(ctx, state)
        new = copy.deepcopy(state)
        bad = np.union1d(dead, noisy)
        cap = int(self.max_mask_frac * (state.ch_weight > 0).sum())
        if len(bad) > cap:                   # keep only the most deviant channels
            rate = ctx.X.mean(0)
            dev = np.abs(np.log((rate[bad] + 1e-4) / (self.ref[bad] + 1e-4)))
            bad = bad[np.argsort(-dev)[:cap]]
        new.ch_weight[bad] = 0
        kept = (new.ch_weight > 0).sum()
        # encoder was trained with masked inputs rescaled by the kept fraction
        new.ch_weight[new.ch_weight > 0] *= (state.ch_weight > 0).sum() / max(kept, 1)
        return new, {"dead": dead.tolist(), "noisy": noisy.tolist(),
                     "thresholds": [self.dead_ratio, self.noisy_ratio]}


class AlignmentAgent:
    name = "alignment"

    def __init__(self, ref_rate, ref_lat_mu, ref_lat_cov, shrink=1.0):
        self.ref, self.mu0, self.C0 = ref_rate, ref_lat_mu, ref_lat_cov
        self.shrink = shrink

    def propose(self, ctx, state, master, attempt=0, mode="input"):
        new = copy.deepcopy(state)
        fit = slice(None)                                # unlabeled: all bins are usable
        if mode == "input":
            rate = (ctx.X * state.ch_weight).mean(0)
            ok = (state.ch_weight > 0) & (rate > 1e-3)
            g = np.ones_like(state.ch_weight)
            g[ok] = np.clip(self.ref[ok] / rate[ok], 0.33, 3.0)
            new.ch_weight = state.ch_weight * g
            return new, {"mode": "input_gain", "median_gain": float(np.median(g[ok])),
                         "max_gain": float(g[ok].max()), "min_gain": float(g[ok].min())}
        # latent CORAL: match mean + covariance of embeddings to day 0
        Z = master.latent(ctx.X, state)[fit]
        mu, C = Z.mean(0), np.cov(Z.T) + 1e-3 * np.eye(Z.shape[1])
        shrink = self.shrink if attempt == 0 else self.shrink * 0.5
        C0 = self.C0
        def msqrt(M, inv=False):
            w, U = np.linalg.eigh(M)
            w = np.maximum(w, 1e-6)
            return (U * (w ** (-0.5 if inv else 0.5))) @ U.T
        A = msqrt(C, inv=True) @ msqrt(C0)
        A = shrink * A + (1 - shrink) * np.eye(len(A))
        new.lat_A, new.lat_mu, new.lat_mu0 = A.astype(np.float32), mu.astype(np.float32), self.mu0.astype(np.float32)
        return new, {"mode": "latent_coral", "shrink": shrink,
                     "mean_shift": float(np.linalg.norm(mu - self.mu0))}


class IntentAgent:
    name = "intent"

    def __init__(self, ref_speed):
        self.ref_speed = ref_speed           # day-0 median decoded speed in movement windows

    def propose(self, ctx, state, master, attempt=0):
        idx, dirs = pseudo_labels(ctx, ctx.fit_trials)
        V = master.decode(ctx.X, state)[idx]
        # orthogonal Procrustes: rotation that best turns decoded velocity toward intent
        sp = np.linalg.norm(V, axis=1, keepdims=True)
        target = dirs * sp
        U, _, Wt = np.linalg.svd(V.T @ target)
        R = U @ Wt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Wt
        gain = self.ref_speed / max(np.median(sp), 1e-6)
        gain = float(np.clip(gain, 0.5, 2.0)) if attempt == 0 else 1.0
        new = copy.deepcopy(state)
        new.out_M = (state.out_M @ R * gain).astype(np.float32)
        return new, {"rotation_deg": float(np.degrees(np.arctan2(R[0, 1], R[0, 0]))), "gain": gain}


class Verifier:
    """Label-free scoring on the verify slice of the day:
       intent_cos  - mean cosine between decoded velocity and pseudo-intent (higher better)
       speed_ok    - decoded speed stays within 0.5-2x of day 0 (guards against collapse)"""

    def __init__(self, ref_speed, margin=0.01):
        self.ref_speed, self.margin = ref_speed, margin

    def score(self, ctx, state, master):
        idx, dirs = pseudo_labels(ctx, ctx.verify_trials)
        V = master.decode(ctx.X, state)[idx]
        sp = np.linalg.norm(V, axis=1)
        cos = float(np.mean((V * dirs).sum(1) / (sp + 1e-6)))
        ratio = float(np.median(sp) / self.ref_speed)
        return {"intent_cos": cos, "speed_ratio": ratio, "speed_ok": 0.5 <= ratio <= 2.0}

    def decide(self, before, after):
        if not after["speed_ok"]:
            return "reject", "decoded speed out of range"
        d = after["intent_cos"] - before["intent_cos"]
        if d > self.margin:
            return "accept", f"intent cosine +{d:.3f}"
        if d > -self.margin:
            return "iterate", f"intent cosine {d:+.3f} within margin"
        return "reject", f"intent cosine {d:+.3f}"


class Orchestrator:
    def __init__(self, master, agents, verifier, max_loops=3, log_path=None):
        self.master, self.agents, self.verifier = master, agents, verifier
        self.max_loops, self.log_path = max_loops, log_path

    def _log(self, rec):
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")

    def plan(self, ctx, state):
        """Pick which agents run this loop, cheapest/most-certain first."""
        steps = []
        if self.agents["channel_health"].needed(ctx, state):
            steps.append(("channel_health", {}))
        steps += [("alignment", {"mode": "input"}), ("alignment", {"mode": "latent"}), ("intent", {})]
        return steps

    def run_day(self, day, ctx, state):
        cur = self.verifier.score(ctx, state, self.master)
        self._log({"day": day, "event": "start", "score": cur, "state": state.summary()})
        tried = set()
        for loop in range(self.max_loops):
            accepted_any = False
            for name, kw in self.plan(ctx, state):
                key = (name, kw.get("mode"))
                if key in tried and name != "channel_health":
                    continue
                agent = self.agents[name]
                for attempt in range(2):        # one iterate allowed per proposal
                    args = (ctx, state) if name == "channel_health" else (ctx, state, self.master)
                    new, info = agent.propose(*args, attempt=attempt, **kw)
                    sc = self.verifier.score(ctx, new, self.master)
                    decision, why = self.verifier.decide(cur, sc)
                    self._log({"day": day, "loop": loop, "agent": name, "kw": kw, "attempt": attempt,
                               "proposal": info, "before": cur, "after": sc,
                               "decision": decision, "why": why})
                    if decision == "accept":
                        state, cur = new, sc
                        state.history.append({"day": day, "agent": name, **kw})
                        accepted_any = True
                        tried.add(key)
                        break
                    if decision == "reject":
                        tried.add(key)
                        break
                else:
                    tried.add(key)
            if not accepted_any:                 # recursion converged
                break
        self._log({"day": day, "event": "end", "loops": loop + 1, "score": cur, "state": state.summary()})
        return state, cur
