# Plan — MTAH 2027 Phase 1: Ball Possession

Solo entry. Machine: MacBook Pro M5, 16 GB. Deadline Tue 17 Nov 2026; internal freeze Fri 14 Nov.

## 1. What the public data already tells us

Public GT (frames 11185-18684): 21 segments = 17 possessions (mean 14.2 s, range 0.76-42.4 s)
+ 4 dead-ball stretches. Confusion of the official `baseline_smooth50` on that window:

| GT \ pred | A | D | H |
|---|---|---|---|
| A (2577) | 1994 | 164 | 419 |
| D (1469) | 253 | **314** | **902** |
| H (3454) | 413 | 0 | 3041 |

The baseline misses most dead ball. Every missed stoppage merges two possessions; every false D
splits one. **Dead-ball detection is the biggest lever.** Short real possessions (<1.5 s) mean
majority-vote smoothing can never be the answer — use duration-aware sequence decoding.

## 2. Pipeline

| stage | does | techniques |
|---|---|---|
| 1 data layer | parse ATD, features, tagging intervals | own parser → npz cache; metres; velocities; attacking direction per half |
| 2 ball reconstruction | fill the 41% missing ball | outlier removal; Kalman / linear fill for short gaps; Ball-Radar-style inference from player positions for long gaps (trained on other games) |
| 3 dead-ball detector | stoppages + restarts | ball out of bounds; stationary ball + collective deceleration; restart formations (kick-off, wall, throw-in, goal kick); Smart Tagging set pieces/goals as soft anchors; Laws-of-the-Game restart team |
| 4 control evidence | per-frame P(H), P(A), P(D) | touch detection (ball direction/speed change near a player); "controlled" checks; team-shape features (spread vs compactness, line heights, pressure on ball); gradient-boosted classifier |
| 5 sequence decoding | probabilities → possessions | hidden semi-Markov model over {H, A, D}, explicit durations, allowed transitions (D→H/A only at restarts), Viterbi; boundary snapping to touches; metric-aware X policy |

## 3. Validation

1. **External games decide everything.** Metrica open sample games (same provider/format),
   then PFF FC World Cup 2022, SkillCorner open data. Derive possession/dead labels from their
   events; degrade the tracking to look like ours (bursty ~40% ball dropout, fragmented tracks,
   IDs stripped); score with the official `possession_metrics.py`.
   **Built 2026-10-30** for the Metrica games: docs/EXTERNAL_VALIDATION.md. Our ~41% missing
   ball turned out to be mostly stoppage time, so the degradation is calibrated to our measured
   tracking profile rather than a flat dropout rate. It reproduces the official full-match
   ranking of the four baselines; its absolute level is optimistic.
2. **Public 5 minutes**: correctness only (alignment, format, label meaning). Never tuning.
3. **Automatic sanity checks**: row count, possession count/duration, all-D pre-kickoff block of
   half 2, ball-gap statistics before/after filling.
4. **Local visual diagnosis** by the human: overlay viewer (video + tracking + predicted labels).
   Failure patterns are described in words and fixed generally — never frame by frame.

## 4. Differentiators (only count if they raise spellF1)

| # | feature | priority |
|---|---|---|
| 1 | dead-ball detector + restart logic | core |
| 2 | team-shape model (signal without the ball) | core |
| 3 | HSMM decoding + metric-aware X policy | core |
| 4 | external validation harness | core |
| 5 | self-training on the target match (algorithmic pseudo-labels) | high |
| 6 | own ball detector on tactical video (YOLO-class, crop around predicted ball, MPS) | stretch |
| 7 | referee-whistle audio, always gated by tracking (crowd whistles!) | stretch, needs audio track |
| 8 | broadcast→tactical sync via formation matching (DTW) | stretch, last |
| 9 | local interactive viewer | support |

Whistle detector rules: soft feature only; tracking must confirm (collective deceleration within
~1-2 s, ball stops); acoustic signature = short burst 0.3-1.5 s, stable pitch ~2.5-4 kHz, clear
harmonics, vs crowd whistling = many overlapping wavering sources.

## 5. Schedule

| dates | work | exit |
|---|---|---|
| Oct 29 - Nov 1 | stage 1 verified; external harness; v1 rule-based | v1 > 0.2433 on external val; submit v1 |
| Nov 2 - Nov 6 | dead ball + restarts; team shape; classifier; HSMM | v2 submitted |
| Nov 7 - Nov 11 | self-training; stretch features as time allows; ablations | keep only what helps |
| Nov 12 - Nov 13 | one-command reproduction; technical document | exact regeneration |
| Fri Nov 14 | **freeze** | |
| Sat Nov 15 | validate exact file, submit | |
| Nov 16 - 17 | buffer | |

## 6. References

- Vidal-Codina et al., *Automatic event detection in football using tracking data* — arXiv:2202.00804
- Kim et al., *Ball Trajectory Inference from Multi-Agent Sports Contexts* (Ball Radar) — arXiv:2306.08206, github.com/pientist/ballradar
- Metrica sample data — github.com/metrica-sports/sample-data
- PFF FC WC 2022 — blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset
- SkillCorner open data — github.com/SkillCorner/opendata
