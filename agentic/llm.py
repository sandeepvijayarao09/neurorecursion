"""Crusoe Managed Inference client (OpenAI-compatible) with a deterministic fallback.

Set CRUSOE_API_KEY to use Nemotron on Crusoe. Without it, callers get `None`
and fall back to their rule-based policy, so the pipeline always runs.
Confirm model IDs with GET /v1/models.
"""
import json
import os
import re
import time

BASE_URL = "https://api.inference.crusoecloud.com/v1"
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B")      # agentic reasoning
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "nvidia/Nemotron-3.5-Lightning-30B-A3B")  # cheap sub-agent

_client = None
CALLS = []   # (role, model, seconds, ok) for the demo readout


KEY_FILE = os.path.expanduser("~/.config/crusoe/api_key")


def _key():
    k = os.environ.get("CRUSOE_API_KEY")
    if not k and os.path.exists(KEY_FILE):
        k = open(KEY_FILE).read().strip()
    return k or None


def available():
    return bool(_key()) and not os.environ.get("NO_LLM")


def _get():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=BASE_URL, api_key=_key())
    return _client


def _extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    return json.loads(m.group(0)) if m else None


def ask_json(role, model, system, user, max_tokens=1500, retries=3):
    """Return a dict parsed from the model's JSON reply, or None if unavailable/failed."""
    if not available():
        return None
    for i in range(retries):
        t = time.time()
        try:
            r = _get().chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0.6,
                messages=[{"role": "system", "content": system + "\nReply with one JSON object only."},
                          {"role": "user", "content": user}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            out = _extract_json(r.choices[0].message.content)
            u = getattr(r, "usage", None)
            CALLS.append((role, model, round(time.time() - t, 1), out is not None,
                          {"id": getattr(r, "id", None), "prompt_tokens": getattr(u, "prompt_tokens", 0),
                           "completion_tokens": getattr(u, "completion_tokens", 0)}))
            if os.environ.get("LLM_LOG"):
                with open(os.environ["LLM_LOG"], "a") as fh:
                    fh.write(json.dumps({"role": role, "model": model, "seconds": CALLS[-1][2], "ok": out is not None,
                                         **CALLS[-1][4], "reply": out}, default=str) + "\n")
            if out is not None:
                return out
        except Exception as e:  # 429/503 are transient per Crusoe; back off
            CALLS.append((role, model, round(time.time() - t, 1), f"error: {type(e).__name__}"))
            time.sleep(2 ** i)
    return None
