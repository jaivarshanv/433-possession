# Experiments

One row per run that matters. External validation decides; the public window is a pipeline check.
Noise: spellF1 differences < 0.04 and macroF1 < 0.01 are not meaningful (the 5–95% bootstrap
interval on the pooled external spellF1 is ~0.05 wide).

External = Metrica sample games 1+2+3, degraded ×3, pooled (759 true possessions per replicate;
`python scripts/12_evaluate_external.py NAME --replicates 3 --bootstrap 1000 --public-check`).
Everything learned is leave-one-game-out. Possessions are per replicate: predicted vs 759 true
(13.7 s). "Ours" = full-match prediction on BAR-ATH (count / mean duration; no labels involved).

## Current reference: harness v2 (12 Nov, ball smoothed + offset like ours, D19)

| id | change | ext spellF1 [5–95%] | ext macroF1 | public spellF1 (check) | pred poss / mean s | ours poss / mean s | notes |
|---|---|---|---|---|---|---|---|
| baseline_smooth50 | official nearest player + ±50 vote | 0.2698 [0.258–0.288] | 0.7189 | 0.3256 | 1625 / 7.5 s | 594 / 7.2 s | official full match 0.2433 / 0.6724 |
| v1_rules | M1: dead-ball rule cells + nearest-player cells with carry, HSMM | 0.5402 [0.501–0.556] | 0.8467 | 0.4875 | 678 / 15.8 s | 247 / 14.7 s | restart team right 0.66 |
| v2_deadball | M2: + restart-like kicks, Laws-of-the-Game restart team (2nd pass), Smart Tagging (ours only) | 0.5533 [0.520–0.573] | 0.8534 | 0.4875 | 696 / 15.5 s | 279 / 13.3 s | restart team 0.76; stoppage ends exact |
| v3_snap | M3: + hand-overs snapped to the first new-team touch after the old team's last | 0.5490 [0.518–0.571] | 0.8535 | 0.5250 | 696 / 15.5 s | 279 / 13.3 s | snapping no longer helps once touches are blurred like ours (D17) |
| v4_gbm | M3: v3 with gradient-boosted P(dead), side-symmetric P(home \| alive) | 0.6778 [0.661–0.715] | 0.9004 | 0.4138 | 635 / 16.7 s | 201 / 17.8 s | |
| v4_gbm_nosnap | v4 without snapping | 0.7221 [0.700–0.754] | 0.9138 | 0.4138 | 635 / 16.7 s | 201 / 17.8 s | snapping costs 0.04 on top of the GBM |
| v4_gbmdead_rulesteam | ablation: GBM dead ball, rule-cell possession | 0.5594 [0.533–0.586] | 0.8609 | 0.4848 | 699 / 15.1 s | 270 / 13.2 s | the GBM gain is in the possession model ... |
| v4_rulesdead_gbmteam | ablation: rule-cell dead ball, GBM possession | 0.7111 [0.692–0.746] | 0.9067 | 0.4444 | 639 / 16.9 s | 202 / 18.7 s | ... not in dead ball |
| v3_snap_x | ablation: v3 + X inside long ball gaps of one possession | 0.6925* | 0.8885* | — | — | — | *harness v1: spellF1 identical by construction (X is forward-filled), macroF1 −0.013 → dropped (D18) |
| **v5_gbm_centred** | **v4_gbm_nosnap with team shape centred on each team's own half-median (style-invariant)** | **0.7178 [0.694–0.751]** | **0.9133** | 0.5143 | 635 / 16.8 s | 206 / 17.4 s | **in `submissions/433.csv`** (D21-D22). Same as v4 within noise; removes a measured team-style bias (home share of live time 0.617 → 0.600). Weights default (13_tune: best +0.005), 1 training replicate (2: +0.009) |

Tuning (13_tune.py, leave-one-game-out): v1 0.629, v2 0.629 (flat), v4_gbm_nosnap 0.7315 LOGO
with all three folds picking (w_dead 0.2, w_team 0.2, w_dur 1.5) -- +0.005 over the defaults,
within noise, so the defaults are kept.

Risk noted (D23): on our match the GBM versions predict ~205 possessions (1.9 / min) against
~270-280 for the rule versions and 2.5-2.8 / min in the external truth. Its confidence on our
match equals its confidence externally; the cause cannot be settled without labels.

## History: harness v1 (30 Oct - 12 Nov, ball pinned to the player as in corrected data)

| date | id | change | ext spellF1 [5–95%] | ext macroF1 | public spellF1 (check) | pred poss / mean s | notes |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | baseline_smooth50 | official nearest player + ±50 vote, through the stage-2 harness | 0.3355 [0.319–0.351] | 0.7463 | 0.3256 | 1447 / 8.4 s | full-match 0.2433 / 0.6724 (published); on our match predicts 594 poss / 7.2 s |
| 2026-11-12 | v1_rules | M1: features + dead-ball rule cells + nearest-player-in-2.5 m cells with carry, probabilities counted on external labels (LOGO), HSMM (log-normal durations, 0.2 s blocks, frame refinement), w_dead = w_team = 0.1 | 0.6331 [0.609–0.664] | 0.8743 | 0.3867 | 698 / 15.5 s | alive/dead F1 0.957. Weights LOGO-tuned: 0.629 (13_tune.py). On our match: 249 poss / 14.5 s (median 9.2 s, none < 1 s), 104 stoppages, D 43% of graded frames; public macroF1 0.776 (check). Harness likely flatters gap-based dead ball (corrected-data gaps align with stoppages) |
| 2026-11-12 | v2_deadball | M2: v1 + restart-like kicks (ball still ≥ 0.4 s then > 4 m/s, or reappearing moving after a ≥ 2 s gap) in the dead-ball and possession cells, heading-out flag, Laws-of-the-Game restart team (corner / goal kick / kick-off after a goal / team at the ball) as a 1 s bonus in a 2nd decoding pass; Smart Tagging bonus on our match only (a priori, ≤ 1 nat) | 0.6445 [0.622–0.674] | 0.8807 | 0.3867 | 724 / 15.0 s | fixes the restart failure mode: restart team right 0.748 → 0.870, stoppage end error median \|·\| 5 → 1 frame (start 3); stoppages 896 pred / 933 true (×3). Rule reliability 0.99 per restart kind. Weights kept at defaults (13_tune: flat, best +0.001). Tagging changes 0 frames at its a-priori weight. On our match: 251 poss / 14.5 s, 107 stoppages |
| 2026-11-12 | v3_snap | M3: v2 + hand-over snapping ("after_last", −0.5 / +6 s) | 0.6868 [0.666–0.718] | 0.8961 | 0.4000 | 724 / 15.0 s | +0.04 on harness v1 -- an artifact of the unrealistically sharp ball (gone on harness v2) |
| 2026-11-12 | v4_gbm | M3: gradient-boosted probabilities | 0.7895 [0.773–0.819] | 0.9370 | 0.5517 | 695 / 15.1 s | led to the harness v2 check: on our match it predicted 209 poss |

## Harness validation: the four official baselines

Same algorithm, four windows; the only systems with published full-match scores.

| baseline | harness v2 spellF1 / macroF1 | harness v1 spellF1 [5–95%] / macroF1 | official full spellF1 / macroF1 | public spellF1 |
|---|---|---|---|---|
| baseline_raw | 0.0492 / 0.7138 | 0.0542 [0.051–0.060] / 0.7398 | 0.0486 / 0.6532 | 0.0378 |
| baseline_smooth25 | 0.1958 / 0.7236 | 0.2510 [0.243–0.267] / 0.7526 | 0.1996 / 0.6752 | 0.2868 |
| baseline_smooth50 | 0.2698 / 0.7189 | 0.3355 [0.319–0.351] / 0.7463 | 0.2433 / 0.6724 | 0.3256 |
| baseline_smooth125 | 0.2960 / 0.6717 | 0.3284 [0.308–0.347] / 0.6923 | 0.2242 / 0.6356 | 0.4242 |

Harness v2 is much closer to the official LEVEL (raw and smooth25 within 0.004) and keeps the
official macroF1 order; smooth50 and smooth125 swap places, a gap inside the noise on both the
harness (0.026) and the official board (0.019).
