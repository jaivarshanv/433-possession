# External validation harness (stage 2)

Every model decision is made here, never on the public 5 minutes (CLAUDE.md §0.2).
The harness takes other games with real event data, derives frame-level `H`/`A`/`D` labels
from those events, degrades their clean tracking until it looks like ours, runs a predictor on
it, and scores the predictor with the official `possession_metrics.evaluate()`.

```bash
python scripts/10_fetch_external.py          # download (sha256-pinned) + derive labels + summary
python scripts/11_profile_tracking.py        # our tracking vs clean vs degraded (aggregates)
python scripts/12_evaluate_external.py baseline_smooth50 --replicates 3 --bootstrap 1000
python scripts/12_evaluate_external.py NAME ... --public-check   # + pipeline check on our match
```

Code: `metrica.py` (download, loaders), `event_labels.py` (labels), `degrade.py` (tracking
degradation), `profile.py` (tracking statistics), `harness.py` (contract + scoring),
`predictors.py` (registry of named predictors). Tests: `tests/test_{metrica,event_labels,
degrade,profile,harness}.py`, synthetic data only.

## 1. Data

[Metrica Sports sample data](https://github.com/metrica-sports/sample-data), pinned to commit
`e706dd5` and checked file by file against sha256 hashes in `metrica.py`. Three anonymised full
matches at 25 fps with **corrected** tracking (every player, ball tracked whenever in play) and
hand-made event data synchronised to it. Games 1–2 are CSV, game 3 is FIFA/EPTS text like our
ATD (with a player↔group mapping that changes after each substitution, handled in the loader).
Pitch 105 × 68 m. Files go to `external/metrica/`, parsed arrays to `external/derived/` (both
gitignored). The repository has no licence file. Its README asks users to use the data
responsibly and credit Metrica Sports.

## 2. Labels from events (`event_labels.py`)

The official GT is "mechanical from corrected tracking": `D` where the corrected tracking marks
the ball dead, otherwise the owning team, which changes only on **controlled** possession. Two
facts measured on the sample games make this reproducible:

- the corrected ball is `NaN` almost exactly while the ball is out of play, and **reappears on
  the frame of the restart** (median offset 0 frames over ~95 restarts per game);
- goal kicks are not `SET PIECE` events but `PASS` / `BALL LOST` with subtype `GOAL KICK`.

Rules (constants named in the module, none tied to a frame):

| | rule |
|---|---|
| D 1 | before each period's kick-off |
| D 2 | a clean-ball `NaN` stretch is a stoppage if a restart is within 1 s of it, a stoppage event (ball out, foul, offside, hand ball, goal, shot out, end of half) happened ≤ 2 s before it or during it, the ball was last seen outside the pitch, it lasts ≥ 10 s, or it runs to the period end. Otherwise it is a tracking hole in live play and stays alive. |
| D 3 | the stoppage starts where the ball crossed the line (walk back over out-of-bounds frames), else where the `NaN` starts; it ends at the restart event, or where the ball reappears |
| D 4 | a restart with no ball gap (a quick free kick, or a 160 s injury stoppage with the ball visible) is `D` from the last stoppage event since the previous restart |
| P 5 | a live stretch starts with the restarting team |
| P 6 | possession changes only at a **controlled** touch of the other team: a restart it takes, the reception of a completed pass, or a recovery / won duel / carry kept ≥ `MIN_HOLD` = 3 frames before the player's next action. A one-touch recovery (Metrica writes `RECOVERY` and `BALL LOST` on the same frame, ~50% of recoveries), a header, a deflection: no change. |

Resulting labels:

| game | graded min | possessions | per min | mean / median s | < 1 s | D share | stoppages (median s) |
|---|---|---|---|---|---|---|---|
| metrica1 | 96.7 | 258 | 2.67 | 12.3 / 8.2 | 5.4% | 45.2% | 102 (21.1) |
| metrica2 | 94.1 | 233 | 2.48 | 14.1 / 9.2 | 5.2% | 41.7% | 105 (16.1) |
| metrica3 | 95.8 | 268 | 2.80 | 14.5 / 9.7 | 3.7% | 32.2% | 104 (12.3) |
| *our public window* | 5.0 | 17 | 3.4 | 14.2 / — | 1 of 17 | 19.6% | 4 |

**Sensitivity to the "controlled" threshold.** With `MIN_HOLD` = 1–3 frames (only same-frame
one-touches are uncontrolled) the four official baselines rank exactly as on the official full
match. Counting every touch (`MIN_HOLD` 0: 1070 possessions, mean 9.7 s) puts smooth25 above
smooth125, which contradicts the official results. Requiring ≥ 0.5 s puts smooth125 above smooth50.
Neighbouring gaps are within noise, so this supports the a-priori choice rather than tuning it.

## 3. Degradation (`degrade.py`)

**The main finding from the profile of our match** (`11_profile_tracking.py`, label-free): our
tracker's missing ball is mostly **stoppage time**, not random dropout in play. It is missing on
80% of frames when the players are nearly still (< 1 m/s mean speed) and on 10% in fast play. It
has 1.4 gaps per minute and no gaps shorter than ~12 frames, and 99.9% of missing frames are in
gaps over 2 s. That is close to the corrected data, where the ball is `NaN` during stoppages.
So a flat "drop 40% of the ball" on top of clean data would make the external games much harder
than ours (~65% missing). Instead every step targets a measured statistic of our match:

| step | what | why (our match) |
|---|---|---|
| 1 | interpolate clean holes < 10 frames | ours has none that short |
| 2 | after the ball leaves play, keep following it while it decelerates, for at most 1.6 s | ours loses the ball when it is **slow** (gap-start hazard 0.0027 at < 5 m/s vs 0.0001 at > 15 m/s); corrected data drops it on the line |
| 3 | in 25% of stoppages show the ball at the restart spot for up to the last 50% | ours sees the ball on ~20% of near-still frames |
| 4 | bursty live-play losses (median 5 s), ×4 on a slow ball, ×3 with ≥ 3 players within 3 m | ours: 10% missing in fast play vs 2–4% clean |
| 5 | each player lost in bursts (median 4 s), ×5 when someone is within 2 m | 20.5 objects per frame, all 22 on only 49% of frames |
| 6 | group "view loss" episodes (10–60% of players, median 2.7 s, ×8 during stoppages) | 0.53 episodes/min with < 18 objects, p1 = 5 objects |
| 7 | identity breaks, ×25 in crowds | 11.2 new tracklets per minute |
| 8 | duplicate tracks 0.3–1.5 m from a real player | > 22 objects on 2.4% of frames |
| 9 | random free slot of the team, pools of 24 / 22 | slots are tracklets, not people |

Calibration (our match vs the three degraded games, aggregates):

| statistic | ours | m1 degr | m2 degr | m3 degr | (clean range) |
|---|---|---|---|---|---|
| ball missing share | 0.407 | 0.395 | 0.405 | 0.327 | 0.32–0.41 |
| ball gaps / min | 1.42 | 1.36 | 1.50 | 1.50 | 1.6–1.8 |
| gap length p10 / p50 / p90 (frames) | 12 / 268 / 845 | 28 / 349 / 1004 | 38 / 199 / 1078 | 41 / 187 / 903 | 1–2 / 127–196 / 859–960 |
| missing when players < 1 m/s | 0.80 | 0.74 | 0.83 | 0.87 | 0.80–0.95 |
| missing when players > 2.5 m/s | 0.10 | 0.11 | 0.09 | 0.05 | 0.02–0.04 |
| gap-start hazard, ball < 5 / > 15 m/s | .0027 / .0001 | .0021 / .0009 | .0027 / .0006 | .0024 / .0005 | ~.001 / ~.004 |
| ball outside the pitch | 2.5% | 3.7% | 2.0% | 3.4% | 0.4–1.7% |
| objects per frame (mean) | 20.49 | 20.43 | 20.54 | 20.90 | 22 |
| frames with all 22 / with ≤ 10 | 49% / 3.3% | 56% / 2.4% | 50% / 2.5% | 57% / 1.8% | 100% / 0 |
| frames with > 22 objects | 2.4% | 1.6% | 1.3% | 1.5% | 0 |
| tracklet starts / min | 11.2 | 12.5 | 11.4 | 11.1 | ~0.3 |
| tracklet length p10 / p50 / p90 (s) | 2.6 / 36 / 284 | 4.3 / 62 / 240 | 4.9 / 69 / 264 | 6.1 / 73 / 278 | whole match |

Remaining differences, known and accepted for now:
- our tracklets have a shorter median (36 s vs 62–73 s);
- in our match fewer objects are tracked while the ball is missing (19.2 vs 21.4), which the
  degraded games reproduce only weakly (≈ 20.4 vs 20.8);
- our player positions are *smoother* than Metrica's corrected tracking (no noise is added);
- the ball-height field is always `NaN` (the clean data has none), so the harness cannot
  validate features that use it;
- team-assignment errors of slots are not simulated (not measurable without labels).

Deterministic: `SEED` + game name + replicate index seed a numpy `default_rng`.

## 4. Scoring (`harness.py`)

Predictor contract: `predict(track, info) -> (N,) codes` where `track` has exactly the keys of
`atd.load_cache()` and `info` holds what the organisers give us for our match (`name`, `pitch`,
`fps`, `half` per frame, `kickoff` per half). `target_inputs()` builds the same two objects for
our match, so a predictor is written once and runs on both.

Each game is scored with the official `evaluate()`. The headline is **pooled**: all games end
to end, separated by ungraded `X` frames, evaluated once (one long "match" with six halves, 759
true possessions). `--replicates R` degrades each game R times. The bootstrap interval uses
replicate 0 only, because replicates share the same true possessions.

## 5. Does the harness rank like the real thing?

The four official baselines (same algorithm, four smoothing windows) are the only systems with
published full-match scores. Degraded ×3, pooled:

| baseline | ext spellF1 [5–95%] | ext macroF1 | clean spellF1 / macroF1 | **official full match** | public window |
|---|---|---|---|---|---|
| raw | 0.054 [0.051–0.060] | 0.740 | 0.062 / 0.828 | 0.0486 / 0.6532 | 0.038 |
| smooth25 | 0.251 [0.243–0.267] | **0.753** | 0.307 / **0.836** | 0.1996 / **0.6752** | 0.287 |
| smooth50 | **0.336** [0.319–0.351] | 0.746 | **0.390** / 0.822 | **0.2433** / 0.6724 | 0.326 |
| smooth125 | 0.328 [0.308–0.347] | 0.692 | 0.357 / 0.750 | 0.2242 / 0.6356 | **0.424** |

- The degraded harness reproduces the official full-match order on **both** metrics
  (spellF1: 50 > 125 > 25 > raw; macroF1: 25 > 50 > raw > 125). The public window picks the
  wrong window (smooth125), as EVALUATION.md warns. Clean tracking gets the macroF1 order wrong
  (raw above smooth50), so the degradation is doing real work.
- **The level is optimistic.** smooth50 scores 0.336 / 0.746 externally vs 0.243 / 0.672 on the
  full match. Use the harness for differences and rankings, not as a forecast of the final
  score. Likely causes: our labels are an approximation of the official GT definition, and some
  difficulties of our tracker are not simulated (see §3).
- Noise: the 5–95% bootstrap interval on pooled spellF1 is about 0.03 wide (759 possessions),
  a bit narrower than the official full-match band of 0.04. Degradation replicates move the
  score by < 0.002.
