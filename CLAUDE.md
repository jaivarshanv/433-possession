# CLAUDE.md — MTAH 2027 Phase 1: Ball Possession

Context for Claude Code working in this repository. Read this fully before doing anything.
**Current work order: `docs/PHASE1_BRIEF.md` (milestones M1-M7 to submission).** Team name: **433**.
Longer background: `docs/PLAN.md`. Open decisions: `docs/DECISIONS.md`. Results log: `docs/EXPERIMENTS.md`.

---

## 0. HARD RULES — read first, never break

### 0.1 Data confidentiality (competition terms)

The competition terms forbid making the match data, **or any extract of it**, available to a
cloud-hosted LLM. You (Claude Code) are a cloud-hosted LLM: anything you read or anything a
command prints to you leaves this machine. Therefore:

- **Never open, read, grep, cat, head, tail, preview, plot-and-view or otherwise look at** anything in
  `data/`, `cache/`, or `submissions/`, or any `*.mp4`, `*_atd.txt`, `*_atd.xml`,
  `*_homography.csv`, `*_smart_tagging.xml` file. A hook in `.claude/hooks/guard_data.py`
  blocks the obvious ways; do not try to get around it.
- **Never view video frames or images rendered from the match** (no screenshots, no Read on PNGs
  made from it). The human looks at visuals locally; you get a verbal description.
- **Scripts you write or run may print aggregates only**: counts, percentages, means, histograms
  as numbers, scores, confusion matrices, possession counts and durations. **Never print
  per-frame rows, coordinate dumps, label sequences, or samples of the raw files.** `df.head()`
  on match data is a violation. If you need to debug a single frame, write the debug output to
  a file under `cache/` for the human to inspect, and print only its path.
- You MAY read everything else: all code, `official/` (public repository), the public 5-minute
  ground truth `official/evaluation/gt/*`, the four committed baselines, and any **external,
  openly licensed** dataset (Metrica sample data, PFF, SkillCorner open data) in `external/`.
- Never commit `data/`, `cache/`, `submissions/`, or `external/` (all gitignored). The public repo
  must not contain match data or frame-level derived data.

### 0.2 Competition rules (RULES.md)

- **Labels must come out of an algorithm.** No hand labels, hand corrections, or rules/constants
  tied to specific frames of this match — not even "frame 51234 is a throw-in".
- **Never train, tune or select between versions using the public 5 minutes**
  (frames 11185-18684). It is for checking the pipeline is *correct* (alignment, format, label
  meaning) only. Model selection and hyperparameters are decided on **external games**.
- No human-produced data about this match from any source (Opta, StatsBomb, Wyscout, LaLiga...).
- Hand labels / event data from **other** games are fine.
- **The repository must regenerate the submitted file exactly.** Every random process is seeded;
  dependencies are pinned; one command rebuilds the submission.

### 0.3 Organisers' rules clarification, 3 Nov 2026 (OVERRIDES everything else)

Not allowed:
- **Checking our output against the video.** No visual verification, no overlay viewer used to
  judge predictions. (Inspecting data/video only to understand what possession means is allowed.)
- Hand labels or any manually created data on the provided game.
- Uploading the provided data to AI services (hence the guard hook; scripts print scores and
  counts only — keep even aggregates minimal).
- Training, fitting or calibrating anything on the ground-truth possession data or the video.

Allowed:
- Automatic checks using only the provided data; files of intermediate steps.
- Using as input the video, the tracking, the Smart Tagging or other generated data.
- Any dataset **we have the rights to use — check its licence**.

Training and tuning MUST use external data only.

---

## 1. The task in one screen

- Match: FC Barcelona (home, `H`) vs Athletic Club (away, `A`), LaLiga 2024-25, matchday 2.
- Output: CSV `frame,label`, exactly **199,319** rows, frames `0..199318` of the **tactical**
  video, labels `H` / `A` / `D` (dead ball) / `X` (abstain).
- Ranked on **spellF1**: possessions (maximal runs of H or of A) matched one-to-one to the truth
  by temporal IoU, F1 averaged over IoU 0.3-0.7. Tie-break **macroF1** (per-frame, H/A/D).
- Baseline floor: spellF1 **0.2433** (nearest player to ball + ±50-frame majority vote).
- Deadline **Tue 17 Nov 2026**. Internal freeze **Fri 14 Nov**.

Key facts that drive design:

| fact | consequence |
|---|---|
| A D stretch splits possessions (`H H D D H H` = 2) | dead-ball detection is the biggest lever |
| Ground truth is mechanical from corrected tracking, unsmoothed; real possessions of 0.76 s exist | don't over-smooth; use duration-aware decoding |
| 1 frame late is free, 8 frames = 0.89, 1 s = 0.65 | boundary timing matters |
| On spellF1, `X` is forward-filled with your previous label in the half | abstain inside occlusions, commit at hand-overs |
| Ball missing on 41% of graded frames, mostly in stoppages (80% when players are still, 10% in fast play) | ball-missing is itself dead-ball evidence; still need ball-free (team-shape) evidence in play |
| No player IDs: 46 track slots, swaps, duplicates | team-level, collective features only |
| Possession changes only on *controlled* possession; loose ball stays with last team | deflections / single touches must not flip the label |
| spellF1 noise ≈ 0.04, macroF1 noise ≈ 0.01 on full match | don't chase small gaps |

## 2. Data facts

- Graded frames (inclusive): half 1 `11185..90651`, half 2 `110808..190197`.
  Half 2 kicks off at `113742`; frames `110808..113741` are graded and all `D`.
- Tracking (ATD) line format: `<frame_id>:<slot>;<slot>;...:<ball>`, groups `x,y,speed`,
  `NaN` when missing. `video_frame = frame_id - 1`. Coordinates normalised 0-1;
  pitch **105.3 × 68.0 m**. Slot → team from the metadata XML (first team = home).
- Smart Tagging: `video_frame = round(seconds * 25)`. 36 instances have Team `N/A`, including
  every throw-in. Phases ≠ possessions. Club names: `Barcelona`, `Atletic Club` (sic).
- Homography: one 3×3 per frame, maps normalised pitch (0-1) → pixels; tactical 100% valid.
- Broadcast: own timeline, NOT synced to tactical; optional.

### Verified on the real bundle (2026-10-30, scripts 00-02)
- Parser VERIFIED: 0 differing frames vs all four official baselines.
- Tactical ATD: 199,319 lines, contiguous, 46 slots (home 24 / away 22). Ball located 59.3% of
  graded frames; 20.5 tracked objects per graded frame (p5 14, p95 22, >22 on only 2.4%).
- 3rd ball field: finite on only 10.5% of frames, median 1.01, p95 3.60 -> almost certainly ball
  HEIGHT in metres, present when the ball is estimated off the ground. Use as an aerial-ball feature.
- Located ball outside the 0..1 pitch box on 2.5% of graded frames -> direct out-of-bounds signal.
- Player speed p50 1.34, p95 4.54, p99.9 8.09 m/s (physically sane).
- Both videos HAVE AUDIO (AAC 48 kHz stereo); tactical audio duration matches video within
  0.07 s -> whistle detection is feasible (decision D6), always gated by tracking.
- Tactical homography 100% valid, reprojection median 0.24. Broadcast 65.8% valid.
- Smart Tagging 638 instances; team H 314 / A 288 / N/A 36; time base agrees with video.

### Tracking profile (2026-10-30, script 11, label-free aggregates)
- Ball gaps: 1.42 per minute, none shorter than ~12 frames (p10), median 268 frames; 99.9% of
  missing frames are in gaps > 2 s. Missing 80% of frames when mean player speed < 1 m/s, 10%
  above 2.5 m/s -> the missing ball is mostly STOPPAGE time, like Metrica's corrected data.
- Gaps start on a SLOW ball (hazard 0.0027/frame below 5 m/s vs 0.0001 above 15 m/s) and in
  crowds (0.0041 with >= 3 players within 3 m vs 0.0013 with none).
- Players: all 22 present on only 49% of frames, <= 10 on 3.3%; 11.2 new tracklets per minute,
  tracklet median 36 s (p10 2.6 s); fewer objects while the ball is missing (19.2 vs 21.4).
  Positions are smoother than Metrica's corrected tracking.

## 3. Repository layout

```
CLAUDE.md                 this file
official/                 the organisers' public repo (git submodule) — scorer, loaders, public GT
data/                     match bundle (gitignored, NEVER read)       tactical/ broadcast/
cache/                    derived arrays, e.g. parsed ATD (gitignored, NEVER read)
submissions/              our output CSVs (gitignored, NEVER read)
external/                 open datasets from other games (gitignored, readable)
src/possession/           our package
  config.py               paths and match constants — the only place they live
  atd.py                  ATD parser -> cache/*.npz ; load_cache()
  frames.py               graded masks, half ids, run-length helpers
  tagging.py              Smart Tagging as a DataFrame with H/A team column
  scoring.py              wrapper around official possession_metrics (any GT, incl. external)
  submission.py           write + validate a submission
  baselines.py            official nearest-player baseline on any tracking dict
  metrica.py              Metrica sample games: pinned download + loaders (external/)
  event_labels.py         external games: frame-level H/A/D labels from their events
  degrade.py              external games: clean tracking -> looks like our ATD (seeded)
  profile.py              aggregate tracking statistics (calibrates degrade.py)
  harness.py              predictor contract; score on degraded external games (official metric)
  predictors.py           registry of named predictors scored by scripts/12; CHOSEN -> 433.csv
  features.py             per-frame ball / player / direction features (metres, team level)
  deadball.py             dead-ball rule cells, CellTable, hysteresis segments
  evidence.py             possession evidence cells, decoder scores, v1 RulesModel
  decoder.py              HSMM over {H, A, D}: durations, transitions, Viterbi, refinement
  training.py             leave-one-game-out fitting on external games (LogoPredictor)
  restarts.py             restart typing + Laws-of-the-Game restart team, v2 DeadballModel
  diagnostics.py          harness stoppage counts, boundary errors, restart-team accuracy
  touches.py              touch detection, hand-over snapping, X policy (ablations), v3 model
  classifier.py           gradient-boosted P(dead) and side-symmetric P(home|alive), v4/v5
scripts/                  numbered pipeline steps; each prints aggregates only
tests/                    pytest, synthetic data only (never match data)
docs/                     PLAN.md, DECISIONS.md, EXPERIMENTS.md, EXTERNAL_VALIDATION.md
```

## 4. Commands

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt
pytest -q                                   # synthetic tests, safe anywhere
python scripts/00_check_data.py             # bundle present? structure, audio track, time base
python scripts/01_build_cache.py            # parse ATD -> cache/ (once, ~1-2 min)
python scripts/02_reproduce_baseline.py     # parser sanity: must reproduce official baseline
python scripts/score.py submissions/x.csv   # validate + score on public window (CORRECTNESS ONLY)
python scripts/10_fetch_external.py         # download Metrica sample games + derive their labels
python scripts/11_profile_tracking.py       # our tracking vs degraded external games (aggregates)
python scripts/12_evaluate_external.py NAME --replicates 3 --bootstrap 1000 [--public-check]
                                            # score registered predictors: THIS decides
python scripts/13_tune.py NAME              # weight grid, leave-one-game-out
python scripts/run.py --predictor NAME [--out submissions/433.csv]   # write + validate
```

A new predictor = a function `predict(track, info) -> codes` (contract in `harness.py`),
registered in `predictors.py`. It runs unchanged on our match via `harness.target_inputs()`.

## 5. Code conventions

- **Simple, heavily commented, beginner-readable code.** Clarity over cleverness. Every function
  gets a docstring saying what goes in, what comes out, and why.
- numpy/pandas first. New dependencies go into `requirements.txt` with an exact pin.
- All paths and constants come from `src/possession/config.py`. Never hard-code a filename.
- Deterministic: fixed seeds, no reliance on dict/set ordering for results.
- Per-frame arrays are indexed by **video frame** (0..199318), length `N_FRAMES`.
- Label codes internally follow the official scorer: `H=0, A=1, D=2, X=3`.
- Every new pipeline stage: a script in `scripts/`, a test in `tests/` on synthetic data, and a
  row in `docs/EXPERIMENTS.md` with its external-validation score.

## 6. Roadmap (update status as you go)

| # | stage | status |
|---|---|---|
| 1 | Data layer: parser, cache, frames, tagging, scoring wrapper | **done, verified 2026-10-30** |
| 2 | External validation harness (Metrica sample games → degraded → labels → official metric) | **done 2026-10-30**; **harness v2 (12 Nov)** adds our tracker's ball smoothing/offset: baseline 0.270 vs 0.243 real (docs/EXTERNAL_VALIDATION.md §6) |
| 3 | v1: ball gap fill + out-of-bounds/stationary dead ball + HSMM over nearest-player evidence | **done 12 Nov** (M1, `v1_rules`): ext spellF1 0.633 [0.609–0.664] vs baseline 0.336; in `submissions/433.csv` |
| 4 | Dead-ball detector + restart logic (Laws of the Game: who restarts) | **done 12 Nov** (M2, `v2_deadball`): ext 0.645 [0.622–0.674], restart team 0.75 → 0.87; in `submissions/433.csv` |
| 5 | Control evidence: touch detection, team-shape features, GBM classifier | **done 12 Nov** (M3, `v5_gbm_centred`, harness v2): ext 0.718 [0.694–0.751] vs v2 0.553; in `submissions/433.csv` |
| 6 | Sequence decoding: HSMM/Viterbi, boundary snapping, X policy | **done 12 Nov** (M4): weights LOGO-tuned (defaults kept), snapping and X policy measured and dropped (D17, D18) |
| 7 | Self-training on target match (algorithmic pseudo-labels) | todo |
| 8 | Stretch: own ball detector (video), whistle audio, broadcast sync | todo |
| 9 | Reproducibility (`make submission`), technical document | todo |

## 7. How to report results to the human

Short. Scores from external validation first, public-window score second (labelled
"pipeline check"). Always show predicted possession count and mean duration next to spellF1.
