"""The committed result pages must regenerate from the committed result files."""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["viz/existing_system.html", "viz/agent_system.html", "viz/evals.html"]


def _strip_timestamp(text):
    return re.sub(r'"generated": "[^"]*"', '"generated": ""', text)


def test_viz_pages_regenerate_identically():
    before = {p: open(os.path.join(ROOT, p)).read() for p in PAGES}
    for script in ["viz/build.py", "viz/build_evals.py"]:
        subprocess.run([sys.executable, os.path.join(ROOT, script)], check=True, cwd=ROOT)
    try:
        for p in PAGES:
            after = open(os.path.join(ROOT, p)).read()
            assert _strip_timestamp(after) == _strip_timestamp(before[p]), f"{p} differs from its result files"
    finally:
        for p, text in before.items():
            open(os.path.join(ROOT, p), "w").write(text)
