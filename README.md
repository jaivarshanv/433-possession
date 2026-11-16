# Team 433 — MTAH 2027 Phase 1: Ball Possession

Our entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(Metrica Sports × FC Barcelona × Mobile World Capital Barcelona): label every frame of
FC Barcelona vs Athletic Club as home possession (`H`), away possession (`A`) or dead ball (`D`).

**This repository contains no match data.** The data is licensed to enrolled participants only.
Technical document: [`docs/TECHNICAL.md`](docs/TECHNICAL.md) (PDF: `docs/433.pdf`).

## Setup

Python 3.11 (developed with 3.11.16), every dependency pinned in `requirements.txt`.

```bash
git clone --recurse-submodules <this repo>
cd <this repo>
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Unpack the match bundle into `data/` (or point `MTAH_DATA_DIR` at it), keeping its file names:

```
data/tactical/BAR-ATH_tactical_atd.xml, _atd.txt, _smart_tagging.xml   (used)
data/tactical/BAR-ATH_tactical.mp4, _homography.csv; data/broadcast/... (not used)
```

## The one command

```bash
python run.py                 # -> submissions/433.csv, validated with the official checks
python run.py --check-repro   # rebuilds twice from empty caches and compares sha256
```

`run.py` parses the tracking into `cache/`, downloads Metrica's open sample games (pinned commit,
sha256-checked, ~180 MB, needs internet the first time) into `external/`, fits the model on
those external games only, predicts every frame of our match and validates the file.

Runtime on the machine we used (MacBook Pro M5, 16 GB, macOS): **~6 minutes on a fresh clone**
(measured: 333 s, almost all of it the one-off ~180 MB sample-data download), **~15 s** on later
runs. The output is byte-identical between runs and machines with the pinned environment
(fixed seeds, pinned versions): a fresh `git clone --recurse-submodules` with only `data/`
linked in reproduced `submissions/433.csv` exactly (sha256 `de445d62…`); `--check-repro` checks
the same locally.

## Tests and the external validation harness

```bash
pytest -q                                          # synthetic data only, ~1 s
python scripts/10_fetch_external.py                # sample games + their event-derived labels
python scripts/11_profile_tracking.py              # our tracking vs the degraded sample games
python scripts/12_evaluate_external.py v5_gbm_centred --replicates 3 --bootstrap 1000
python scripts/13_tune.py v4_gbm_nosnap            # decoder weights, leave-one-game-out
python scripts/run.py --predictor v2_deadball      # any registered version -> submissions/
```

Every model choice was made on the external harness (`docs/EXTERNAL_VALIDATION.md`), never on
the public five minutes. Results: `docs/EXPERIMENTS.md`; decisions: `docs/DECISIONS.md`.

## Layout

```
run.py                one command: data bundle -> submissions/433.csv
src/possession/       the package (CLAUDE.md §3 lists every module)
scripts/              numbered steps; every script prints aggregates only
tests/                pytest on synthetic data
docs/                 TECHNICAL.md (433.pdf), EXTERNAL_VALIDATION.md, EXPERIMENTS.md, DECISIONS.md
official/             the organisers' repository (git submodule): scorer, loaders
```

## Credits and licence

External training/validation data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data)
(three anonymised matches, tracking and event data by Metrica Sports). Our code: MIT (`LICENSE`).
