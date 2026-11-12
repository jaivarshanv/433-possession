# MTAH 2027 — Phase 1: Ball Possession

Our entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(Metrica Sports × FC Barcelona × Mobile World Capital Barcelona): label every frame of
FC Barcelona vs Athletic Club as home possession, away possession or dead ball.

**This repository contains no match data.** The data is licensed to enrolled participants only.

## Setup

```bash
git clone --recurse-submodules <this repo>
cd <this repo>
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Unpack the match bundle into `data/` (or set `MTAH_DATA_DIR`):

```
data/tactical/BAR-ATH_tactical.mp4, _atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
data/broadcast/BAR-ATH_broadcast.mp4, _atd.xml, _atd.txt, _homography.csv
```

## Run

```bash
python scripts/00_check_data.py           # verify the bundle
python scripts/01_build_cache.py          # parse tracking once
python scripts/02_reproduce_baseline.py   # sanity check against the official baseline

# external validation (other games, openly available): decides every model choice
python scripts/10_fetch_external.py       # download Metrica's sample games, derive labels
python scripts/12_evaluate_external.py baseline_smooth50 --replicates 3 --bootstrap 1000
```

The full submission command will be documented here once the pipeline is complete.

## Layout

See `CLAUDE.md` §3. Approach and plan: `docs/PLAN.md`.
