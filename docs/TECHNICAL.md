# Team 433 — Ball possession from automatic tracking (MTAH 2027, Phase 1)

Every frame of FC Barcelona – Athletic Club is labelled `H`, `A` or `D` by an algorithm that
was built, tuned and selected **only on other matches**. Code: this repository (`python run.py`).

## 1. Approach, and why

Possessions are scored as whole intervals (spellF1), so we predict **segments**, not frames:

1. **Features** (`features.py`): per frame, in metres, team level only (slots are tracklets, not
   people): ball position, speed, acceleration, gap length, distance outside the pitch,
   "restart-like kicks" (ball still, then struck), team speeds, centroids, spread, extent,
   nearest home/away player to the ball, ball touches, attacking direction (read from the
   kick-off formation).
2. **Per-frame probabilities**: P(ball dead) and P(home has it | alive), from two gradient-boosted
   models (scikit-learn `HistGradientBoostingClassifier`). The possession model is exactly
   side-symmetric (trained on every frame twice, as seen and with the teams swapped; prediction =
   average of both views), and its team-shape inputs are centred on each team's own half-median
   so that a team's *style* (wide, high) is not read as possession.
3. **Restarts by the Laws of the Game** (`restarts.py`): after each stoppage, the restart is
   typed by where the ball is struck — corner (to the attackers of that goal line), goal kick (to
   the defenders), kick-off after a goal (to the conceding team), otherwise the team at the ball —
   and that team gets strong evidence for the first second after the restart.
4. **Hidden semi-Markov decoding** (`decoder.py`) over {H, A, D}: log-normal durations and
   side-symmetric transitions (H↔A, H/A→D, D→H/A) fitted on external labels; Viterbi on 0.2 s
   blocks with full-frame boundary refinement; two passes (find stoppages, add restart evidence).

Why: dead ball splits possessions and is the biggest lever; per-frame decisions flicker; a
semi-Markov model makes a 0.2 s "possession" expensive while still allowing real 1 s ones.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical tracking (ATD), players + ball (x, y) | the only real input |
| Smart Tagging | SET PIECES team as a tiny fixed bonus (≤ 1 nat per phase, chosen a priori, cannot be tuned: the external games have no tagging). At that weight it changes **0 frames**: effectively unused |
| ball height (3rd ball field) | dropped: the external games have none, so it cannot be validated |
| homography, tactical video, audio, broadcast | dropped: tracking is already in pitch coordinates; no video model was built; our output was never compared with the video |
| public 5-minute ground truth | pipeline correctness check only (format, alignment); never used to choose or tune |

## 3. Training data

[Metrica Sports' open sample data](https://github.com/metrica-sports/sample-data) (pinned commit
`e706dd5`): three anonymised full matches with corrected tracking and **event data produced by
Metrica Sports**. We derive frame labels from those events (`event_labels.py`): dead ball from
the ball crossing the line / the whistle to the restart (Metrica's corrected ball vanishes almost
exactly while out of play and reappears at the restart); possession changes at the other team's
controlled touch. Amount: 3 × ~95 minutes, 759 possessions, 311 stoppages. The tracking is
**degraded** to look like ours (`degrade.py`: bursty ball losses, ball smoothed and offset from
the players, players lost in bursts and groups, identity switches, duplicates, random slots —
each calibrated on label-free aggregates of our tracking). **Nothing from the provided game** is
used for training, tuning or selection: no labels, no ground truth, no video.

## 4. Possession rules implemented

- **Loose ball**: stays with the last team in control.
- **Control**: a completed pass reception, a restart taken, or a recovery / won duel / carry kept
  ≥ 3 frames (0.12 s). A one-touch recovery (recovery and loss on the same frame), a header won
  without control, a deflection: no change of possession.
- **Challenges**: count only if the winner then keeps the ball.
- **Dead ball**: from the line crossing (or the whistle) to the restart; before each kick-off.
- **Restarts**: Laws of the Game as in §1.3.
- **X policy**: none. The scorer forward-fills X with the previous label, so X inside a possession
  cannot change spellF1 and only costs macroF1 (measured: identical spellF1, −0.013 macroF1).

## 5. Missing and noisy tracking

The ball is missing on 41% of graded frames, but a profile of our tracking showed it is mostly
**stoppage time** (80% missing when players stand still, 10% in fast play), so gap structure is
itself dead-ball evidence. No player identities are used (46 slots for 22 players): only team
counts, team means and the nearest player of each team. Our tracker's ball is smoothed and offset
from the players; the simulator reproduces this (harness v2), which also showed that snapping
hand-overs to detected touches only helped on unrealistically clean data — so it was dropped.

## 6. External validation and ablations

Pooled over the 3 degraded games × 3 degradation seeds, leave-one-game-out, official
`possession_metrics.evaluate()`; 5–95% bootstrap interval over possessions.

| version | change | spellF1 [5–95%] | macroF1 | poss / game (true 253) |
|---|---|---|---|---|
| official baseline (smooth50) | nearest player + ±50-frame vote | 0.270 [0.258–0.288] | 0.719 | 542 |
| v1_rules | rule cells + HSMM | 0.540 [0.501–0.556] | 0.847 | 226 |
| v2_deadball | + restart kicks, Laws-of-the-Game restart team | 0.553 [0.520–0.573] | 0.853 | 232 |
| v3_snap | + hand-overs snapped to touches | 0.549 [0.518–0.571] | 0.854 | 232 |
| v4_gbm_nosnap | gradient-boosted probabilities | 0.722 [0.700–0.754] | 0.914 | 212 |
| **v5_gbm_centred (submitted)** | + style-invariant team shape | **0.718 [0.694–0.751]** | **0.913** | 212 |

Ablations: GBM for dead ball only 0.559; GBM for possession only 0.711 (the gain is in the
possession model); snapping on top of the GBM −0.04; X policy ±0 spellF1, −0.013 macroF1;
decoder weights grid (leave-one-game-out) best +0.005 → defaults kept. The harness reproduces
the official full-match levels of the published baselines (smooth50 0.270 vs 0.243; raw and
smooth25 within 0.004) and their macroF1 order. Public 5 minutes (check only): v5 spellF1 0.514,
macroF1 0.726. On our match v5 predicts 206 possessions (mean 17.4 s) and 113 stoppages.

## 7. Limitations

- Three external games only (≈ ±0.03 on spellF1); our derived labels approximate, but are not,
  the official ground-truth definition.
- Domain shift: models are trained on simulated degradation. On our match the GBM versions
  predict ~205 possessions (1.9 / min) against ~270–280 for the rule versions and 2.5–2.8 / min in
  the external truth — possibly under-segmentation that the harness cannot see.
- Possessions under ~1 s are almost never predicted (the duration prior suppresses them).
- With more time: more external games (PFF FC, SkillCorner) for training volume, ball height,
  and a learned sequence model instead of fixed log-normal durations.

## 8. Reproduction

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt
python run.py                # data/ -> submissions/433.csv (~15 s on a MacBook Pro M5, 16 GB)
python run.py --check-repro  # two rebuilds from empty caches: identical sha256
pytest -q                    # synthetic tests
```

External data credit: tracking and event data of the three sample matches by **Metrica Sports**
(github.com/metrica-sports/sample-data).
