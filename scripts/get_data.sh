#!/usr/bin/env bash
# Download the public datasets used by the demos (not stored in git).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/mc_maze data/falcon_h1/held_in data/falcon_h1/held_out data/raw
# MC_Maze medium (DANDI 000139), ~77 MB
curl -sL -o data/mc_maze/mc_maze_medium_train.nwb \
  "https://api.dandiarchive.org/api/assets/7ef450a8-8684-42e2-8598-cd38ca2b2e50/download/"
# FALCON H1 calibration sessions (DANDI 000954), ~90 MB
curl -s "https://api.dandiarchive.org/api/dandisets/000954/versions/draft/assets/?page_size=100" | python3 -c '
import sys, json
for a in json.load(sys.stdin)["results"]:
    p = a["path"]
    if "minival" in p:
        continue
    d = "held_in" if "held-in" in p else "held_out"
    print(d + "/" + p.split("_ses-")[1], a["asset_id"])' | while read -r f id; do
  curl -sL -o "data/falcon_h1/$f" "https://api.dandiarchive.org/api/assets/$id/download/"
done
# Neuralink 2024 compression-challenge recordings (official URL is offline; community mirror)
curl -sL -o data/data.zip "https://github.com/mikaelhaji/n1-codec/raw/main/data.zip"
unzip -q -o data/data.zip -d data/raw
echo "done"
