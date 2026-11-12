# Experiments

One row per run that matters. External validation decides; the public window is a pipeline check.
Noise: spellF1 differences < 0.04 and macroF1 < 0.01 are not meaningful.

External = Metrica sample games 1+2+3, degraded ×3, pooled (759 true possessions per replicate;
`python scripts/12_evaluate_external.py NAME --replicates 3 --bootstrap 1000 --public-check`).
External scores run optimistic in level: rank with them, don't forecast with them
(docs/EXTERNAL_VALIDATION.md §5). Possessions are per replicate: pred / gt 759 (13.7 s).

| date | id | change | ext spellF1 [5–95%] | ext macroF1 | public spellF1 (check) | pred poss / mean s | notes |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | baseline_smooth50 | official nearest player + ±50 vote, through the stage-2 harness | 0.3355 [0.319–0.351] | 0.7463 | 0.3256 | 1447 / 8.4 s | full-match 0.2433 / 0.6724 (published); on our match predicts 594 poss / 7.2 s |
| 2026-11-12 | v1_rules | M1: features + dead-ball rule cells + nearest-player-in-2.5 m cells with carry, probabilities counted on external labels (LOGO), HSMM (log-normal durations, 0.2 s blocks, frame refinement), w_dead = w_team = 0.1 | 0.6331 [0.609–0.664] | 0.8743 | 0.3867 | 698 / 15.5 s | alive/dead F1 0.957. Weights LOGO-tuned: 0.629 (13_tune.py). On our match: 249 poss / 14.5 s (median 9.2 s, none < 1 s), 104 stoppages, D 43% of graded frames; public macroF1 0.776 (check). Harness likely flatters gap-based dead ball (corrected-data gaps align with stoppages) |
| 2026-11-12 | v2_deadball | M2: v1 + restart-like kicks (ball still ≥ 0.4 s then > 4 m/s, or reappearing moving after a ≥ 2 s gap) in the dead-ball and possession cells, heading-out flag, Laws-of-the-Game restart team (corner / goal kick / kick-off after a goal / team at the ball) as a 1 s bonus in a 2nd decoding pass; Smart Tagging bonus on our match only (a priori, ≤ 1 nat) | 0.6445 [0.622–0.674] | 0.8807 | 0.3867 | 724 / 15.0 s | fixes the restart failure mode: restart team right 0.748 → 0.870, stoppage end error median \|·\| 5 → 1 frame (start 3); stoppages 896 pred / 933 true (×3). Rule reliability 0.99 per restart kind. Weights kept at defaults (13_tune: flat, best +0.001). Tagging changes 0 frames at its a-priori weight. On our match: 251 poss / 14.5 s, 107 stoppages |

## Harness validation: the four official baselines

Same algorithm, four windows; the only systems with published full-match scores. The harness
reproduces the official full-match ranking on both metrics; the public window does not.

| baseline | ext spellF1 [5–95%] | ext macroF1 | pred poss / mean s | official full spellF1 / macroF1 | public spellF1 |
|---|---|---|---|---|---|
| baseline_raw | 0.0542 [0.051–0.060] | 0.7398 | 6222 / 1.7 s | 0.0486 / 0.6532 | 0.0378 |
| baseline_smooth25 | 0.2510 [0.243–0.267] | 0.7526 | 2041 / 5.7 s | 0.1996 / 0.6752 | 0.2868 |
| baseline_smooth50 | 0.3355 [0.319–0.351] | 0.7463 | 1447 / 8.4 s | 0.2433 / 0.6724 | 0.3256 |
| baseline_smooth125 | 0.3284 [0.308–0.347] | 0.6923 | 1054 / 12.9 s | 0.2242 / 0.6356 | 0.4242 |
