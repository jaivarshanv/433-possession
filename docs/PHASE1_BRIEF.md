# Phase 1 brief — finish the possession system and submit

Team **433** (registered, 3+ members). One person is building; you (Claude Code) write the code.
This brief is the work order from now to submission. Re-read it at the start of every session.

**SCHEDULE REVISED 12 Nov** (nothing was built 30 Oct - 12 Nov). M5 and M6 are DROPPED.

| when (IST) | milestone |
|---|---|
| Thu 12 - Fri 13 Nov | **M1** v1 → valid `submissions/v1.csv`, first git commit, public GitHub repo created |
| Fri 13 - Sat 14 Nov | **M2** dead ball + restart logic (biggest lever) → v2 |
| Sat 14 - Sun 15 Nov | **M3-lite** controlled touches + team-shape evidence (GBM only if time) + **M4** tuning → v3 |
| Mon 16 Nov | **M7** reproducibility, technical document `433.pdf`, repo final. **Freeze Mon 16 Nov 20:00 IST** |
| Tue 17 Nov | submit via the organisers' form (sent by 14 Nov). Official deadline Tue 17 Nov 23:59 CET = **Wed 18 Nov 04:29 IST** |

Rule of the week: **always have a valid, committed submission on disk.** Each milestone ends with
a new `submissions/vN.csv` only if it beats the previous one on the external harness.

---|---|
| Thu 5 Nov | **M1** v1 submission file exists and beats baseline on external validation |
| Mon 9 Nov | **M2** dead-ball + restart logic, **M3** classifier |
| Wed 11 Nov | **M4** decoder tuned, **M5** self-training (keep only if it helps) |
| Thu 12 Nov | **M6** stretch, strictly time-boxed (whistle audio) |
| Fri 13 Nov | **M7** reproducibility, public repo, technical document |
| **Sat 14 Nov** | **code freeze** — organisers send the submission form this day |
| Tue 17 Nov 23:59 CET | official deadline (= Wed 18 Nov 04:29 IST). We submit on the 15th. |

---

## 0. Non-negotiable rules (repeat of CLAUDE.md §0 — they override everything below)

1. **Never look at match data.** Not `data/`, `cache/`, `submissions/`, not the videos, not frames.
   Scripts print **aggregates only** (counts, rates, percentiles, scores, confusion matrices,
   possession counts/durations). No `head()`, no per-frame dumps, no label sequences. If a human
   needs to inspect something frame-level, write it to `cache/debug/` and print only the path.
2. **Never tune or select on the public 5 minutes** (frames 11185–18684). Every hyperparameter,
   threshold and model choice is decided on the **external harness**
   (`scripts/12_evaluate_external.py`). The public window is only printed as a correctness check
   (`--public-check`), labelled as such, and never used to choose.
3. **No frame-specific rules or constants** for our match. Every rule must be general football
   logic that would work on any other match.
4. **Smart Tagging is evidence, never labels.** It may shift probabilities or anchor restarts; it
   may not be copied into the output.
5. **Determinism.** Fixed seeds, pinned dependencies, sorted iteration. Running the pipeline twice
   must produce byte-identical CSVs (we check the sha256).
6. **Simple, heavily commented, beginner-readable code.** Docstrings say what goes in, what comes
   out, and why. Prefer clear numpy over clever tricks.
7. Every change that could affect the score → one row in `docs/EXPERIMENTS.md`
   (external spellF1 with 5–95% interval, macroF1, predicted possessions / mean duration,
   public-window check). A change is **kept only if external spellF1 improves by more than the
   noise band (~0.02–0.03) or it fixes a known failure mode without hurting the score.**
8. Commit after every milestone (`git add` only code/docs; run `git status` and confirm no data,
   cache, submissions or external files are staged).

## 1. What we know (read docs/EXTERNAL_VALIDATION.md for detail)

- Ranking = spellF1: whole possessions matched by IoU. A dead-ball stretch **splits** a
  possession; missing one merges two, a false one splits one. **Dead ball is the biggest lever.**
- On the public window the baseline gets only 314 / 1469 dead frames right (902 called `H`).
- Real possessions can be 0.76 s long; the GT is unsmoothed → duration-aware decoding, not
  big majority windows.
- **Our tracker loses the ball mostly during stoppages**: missing on 80% of frames when players
  are nearly still (< 1 m/s mean), 10% in fast play; gaps are long (median 268 frames), 1.4/min.
  So "ball missing + players slow" is a strong dead-ball signal.
- Ball outside the 0..1 pitch box on 2.5% of graded frames → out-of-play signal.
- 3rd ball field = ball height (m), finite on ~10% of frames. **Not present in the external
  games**, so the harness cannot validate it: use it only in rule-based parts, and only if it
  is neutral-or-better when ablated.
- Possession changes only on **controlled** possession (reception of a completed pass, a
  recovery kept ≥ a few frames, a restart taken). Deflections, single touches, headers won
  without control: no change. While loose, possession stays with the last team.
- Second half: graded from 110808, kick-off at 113742 → frames before are all `D`.
  First half graded from its kick-off 11185.
- X on spellF1 = "repeat my previous label". Abstaining inside a possession is free; across a
  hand-over it is costly.
- Harness level is optimistic (baseline 0.336 external vs 0.243 real): use it to **rank**.

## 2. Architecture

```
track (atd cache) ──► features.py ──► per-frame feature table (both our match and external games)
                                   │
            deadball.py  ──────────┤  P(dead) per frame + stoppage segments + restart frames
            restarts.py  ──────────┤  which team restarts each stoppage (Laws of the Game)
            touches.py   ──────────┤  touch events: frame, team, controlled?
            classifier.py ─────────┤  GBM → P(H), P(A), P(D) per frame
                                   ▼
            decoder.py   ── HSMM / Viterbi over {H, A, D}, durations + allowed transitions,
                                   boundary snapping, X policy
                                   ▼
            predictors.py  registry entry  →  harness (external)  /  run.py (our match)
```

Every module works on the predictor contract already in `harness.py`
(`predict(track, info) -> codes`), so one implementation runs on both the external games and
our match. Add each predictor version to `predictors.py` under a clear name (`v1_rules`,
`v2_deadball`, `v3_gbm`, ...), never overwrite an old one: the experiment log must stay
reproducible.

---

## M1 — v1: dead ball + nearest player + HSMM (target Thu 5 Nov)

Goal: a complete, valid `submissions/v1.csv` that beats `baseline_smooth50` on external
spellF1 by a clear margin. Simple rules, no learning yet.

1. `features.py` — per frame, in metres, team-level only (slots are tracklets, not people):
   - ball: present, x, y, speed and acceleration from finite differences (only across present
     frames, NaN otherwise), current gap length and position inside the gap, distance outside
     the pitch box (0 if inside), out-of-bounds flag;
   - players: number tracked per team, mean / p75 speed of all players and per team (rolling
     1 s), team centroids, nearest home and nearest away player distance to ball (when present);
   - attacking direction of each team per half, inferred from data (e.g. goalkeeper-like
     deepest slot / team centroid in the first minutes after kick-off) — never hard-coded.
2. `deadball.py` v1 — score in [0,1] combining: ball out of bounds (strong), ball missing for a
   long gap **and** players slow (strong), ball present but static + players slow (medium),
   before kick-off of each half (certain). Then turn into segments with hysteresis
   (enter / exit thresholds, minimum length) — thresholds chosen on the harness.
3. Possession evidence v1 — nearest player team with a distance gate (only count frames where
   the nearest player is within ~1.5–2 m, else "unknown"), plus "last known team" carried
   through unknown frames.
4. `decoder.py` — hidden semi-Markov model over {H, A, D}:
   - duration distributions: fit from the **external labels** (log-normal or a histogram) for
     possession and stoppage lengths;
   - transitions: H↔A, H/A→D, D→H/A; no D→D, no self-loops (durations handle those);
   - emissions from the evidence above as log-probabilities with a few weights;
   - exact Viterbi in a segment-length-limited form (cap duration, e.g. 120 s, use a coarse
     time step such as 5 frames for speed, then refine boundaries at full rate). Must run on the
     whole match in < ~2 minutes on a MacBook M5.
5. `scripts/20_run_v1.py` (or a generic `scripts/run.py --predictor NAME`): predict our match,
   write `submissions/v1.csv` via `submission.write_submission` (it validates), print
   aggregates: label shares, possession count / mean / median duration, stoppage count.
6. Evaluate: `scripts/12_evaluate_external.py v1_rules --replicates 3 --bootstrap 1000
   --public-check`. Log it. Report to the human.

Sanity numbers to watch (not targets to tune toward): external GT has ~2.5–2.8 possessions /
min, mean ~13 s; our public window shows 17 in 5 min. The baseline predicts 594 possessions on
our match (median 4.6 s) — far too many.

## M2 — dead-ball detector and restart logic (target Mon 9 Nov)

1. Stoppage **start** precision: walk back to where the ball crossed a line (last present
   positions extrapolated to the boundary), or where the ball/players decelerated together.
   Stoppage **end** (restart): ball reappears and accelerates, a player is at the ball and
   others are set (throw-in: player on touchline; corner: ball at corner arc; goal kick: ball
   in goal area, opponents outside the box; free kick: wall / stationary ball; kick-off: both
   teams in own halves, ball at centre spot).
2. `restarts.py` — who restarts, from the Laws of the Game:
   - over the touchline → throw-in to the team that did **not** touch last;
   - over the goal line, last touch by attackers → goal kick to defenders; by defenders →
     corner to attackers (use attacking direction);
   - after a goal → kick-off by the conceding team (goals: ball crosses goal line inside the
     posts, Smart Tagging `GOALS` as soft evidence);
   - otherwise (foul/offside, ball stays on pitch): team nearest the ball at restart with
     the opponents ≥ 9 m away; Smart Tagging `SET PIECES` team as soft evidence (note throw-ins
     are tagged `N/A` — our rule fills that gap).
   Feed "restart team" into the decoder as strong evidence for the first seconds after D.
3. Smart Tagging: map each instance to soft evidence (set piece intervals raise P(D) just before
   their start and indicate the restarting team; phases with a team tag raise that team's
   prior). The external games have **no Smart Tagging**, so its weight cannot be tuned on the
   harness: give it a small fixed weight chosen a priori, document the choice, and use the public
   window only to confirm it does not **break** anything (never to set the weight).
4. Measure separately on the harness: alive/dead F1, stoppage count error, median start/end
   boundary error in frames. Log them.

## M3 — control evidence and classifier (target Mon 9 Nov)

1. `touches.py` — touch = ball direction change > θ or speed change > δ with a player within
   ~1.5 m; team = that player's slot team. Controlled = same team keeps the ball (nearest within
   gate) ≥ N frames, or two consecutive same-team touches, or a completed pass (next touch by a
   teammate). Thresholds set on the harness.
2. Team-shape features (signal when the ball is missing): per team width, length, convex-hull
   area, centroid x relative to attacking direction, defensive-line height, compactness,
   number of players within 5/10 m of the (last known) ball, team mean velocity along the
   attacking axis, difference home − away for each. Rolling means over 1 s / 3 s.
3. `classifier.py` — gradient-boosted trees (`scikit-learn` `HistGradientBoostingClassifier`,
   pin the version in requirements.txt) predicting H / A / D per frame (subsample, e.g. every
   5th frame, for training). **Train only on degraded external games.** Leave-one-game-out
   cross-validation; train on several degradation seeds as augmentation. Make features
   side-symmetric (express things as "team in attacking direction" and also mirror the pitch)
   so the model does not learn "home = H" shortcuts. Calibrate probabilities (isotonic on
   held-out game).
4. Feed classifier log-probabilities into the HSMM as emissions (replacing / combining with the
   M1 hand-made emissions). Compare v2 (rules) vs v3 (GBM) on the harness.

## M4 — decoding refinements (target Wed 11 Nov)

1. Tune HSMM weights and duration priors with a small grid search on the harness, using
   leave-one-game-out (choose on two games, check the third). Report the chosen values and
   the spread; prefer the simplest setting within noise of the best.
2. Boundary snapping: move each H↔A transition to the nearest controlled touch within ±1 s;
   move D boundaries to the detected line-crossing / restart frames.
3. X policy: inside a long ball-missing stretch that the decoder keeps in one possession, emit
   `X` (free on spellF1, protects against splitting); never emit X across a predicted
   transition. Check its effect on spellF1 and macroF1 separately (macroF1 pays for X, it is the
   tie-break — keep X only where it gains spellF1).

## M5 — DROPPED (12 Nov, no time). Kept for reference only

Allowed by RULES.md ("pseudo-labels produced by an algorithm"). Predict our match, keep frames
with high confidence (and away from boundaries), add them to training with a lower weight,
retrain the classifier, re-decode. Simulate first on the harness (treat one external game as
"target", pseudo-label it, check that its score improves). Keep only if the harness says so.
Document exactly what was done.

## M6 — DROPPED (12 Nov, no time). Kept for reference only

Only if M1–M5 are done. Referee whistle from the tactical video audio:
`ffmpeg` → mono 16 kHz WAV in `cache/` (never shown to the model); detect short (0.3–1.5 s)
narrow-band bursts at 2–4.5 kHz with harmonics; output whistle candidate frames as a **soft**
dead-ball feature that only counts when tracking agrees (collective deceleration within 2 s).
Crowd whistles are common: never let audio alone create a stoppage. Validation is limited (no
external audio): report how many candidates coincide with tracking-detected stoppages, and only
enable it if it changes few frames and does not break the public window check.
If M1–M5 slipped, **skip M6** and go to M7.

## M7 — reproducibility, repository, technical document (target Fri 13 Nov)

1. One command regenerates the submission from the data bundle:
   `python run.py` (or `make submission`) → `submissions/433.csv`, validated. Run it twice from
   a clean `cache/` and compare sha256: must be identical. Pin every dependency in
   `requirements.txt`; record Python version.
2. README: setup, data placement, the one command, expected runtime, hardware used
   (MacBook Pro M5, 16 GB), how to run tests and the external harness.
3. Repository hygiene before going public: `git ls-files` must contain **no** data, cache,
   submissions, external datasets, videos or frame-level derived files. The `official/`
   submodule stays a submodule. Add a LICENSE (MIT) for our code.
4. `docs/TECHNICAL.md` (→ exported to `433.pdf`, a few pages), exactly the sections SUBMISSION.md
   asks for: approach and why; assets used and dropped (with reasons); training data (Metrica
   open sample games: source, who labelled, how much; our derived labels; **nothing from the
   provided game**); possession rules implemented (loose balls, challenges, restarts, X
   policy); handling of missing/noisy tracking; limitations; reproduction. Include the external
   validation table and the ablations from EXPERIMENTS.md.
5. Final checks: `python official/evaluation/validate_submission.py submissions/433.csv`, file
   named `433.csv`, document `433.pdf`, repo public URL works in a private browser window.

## 3. How to report to the human after each milestone

Short block: what was built (one line per module), external spellF1 [5–95%] and macroF1 vs the
previous best and vs baseline_smooth50, predicted possession count / mean duration on our
match, public-window check (labelled "check only"), decisions made and why, what is next, and
anything that needs a human decision. Do not paste code unless asked.
