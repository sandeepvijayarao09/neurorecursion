"""Check the Crusoe connection: list Nemotron model IDs and send one JSON ping per role.
Never prints the key. Usage: python check_crusoe.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm  # noqa: E402

if not llm.available():
    sys.exit(f"No key. Save it with:  mkdir -p ~/.config/crusoe && read -rs K && printf %s \"$K\" > {llm.KEY_FILE} && chmod 600 {llm.KEY_FILE}")
ids = [m.id for m in llm._get().models.list().data]
nem = [i for i in ids if "emotron" in i.lower()]
print("Nemotron models on this key:", *nem, sep="\n  ")
for role, model in [("planner", llm.PLANNER_MODEL), ("reviewer", llm.REVIEWER_MODEL)]:
    if model not in ids:
        close = [i for i in nem if model.split("/")[-1].lower()[:14] in i.lower()]
        print(f"{role}: {model} NOT FOUND. Closest: {close}. Set {role.upper()}_MODEL=<id>")
        continue
    out = llm.ask_json(role, model, "You are a health check.", 'Reply {"ok": true, "model_says": "<5 words>"}', max_tokens=200)
    print(f"{role}: {model} ->", json.dumps(out), llm.CALLS[-1][2], "s")
