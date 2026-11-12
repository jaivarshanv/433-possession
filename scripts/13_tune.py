"""Choose decoder weights on the external harness, leave-one-game-out. Aggregates only.

    python scripts/13_tune.py v1_rules
    python scripts/13_tune.py v1_rules --replicates 1

For a registered LOGO predictor, every model is fitted once per held-out game (training.py).
Then every combination of the weight grid below is decoded on every game, and:

  * IN-SAMPLE table: pooled spellF1 of each setting over all games (what a plain grid search
    would report -- optimistic, shown for reference)
  * LEAVE-ONE-GAME-OUT: for each game, pick the best setting on the OTHER games, score it on
    this one. The pooled LOGO score is the honest estimate of "tune, then predict a new game".

Decision rule printed at the end: KEEP the a-priori default setting unless some setting beats
it in-sample by more than NOISE (then that setting, and it must also look good leave-one-game-
out). Nothing is written automatically.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import features, harness, predictors, scoring  # noqa: E402

NOISE = 0.02        # spellF1 differences below this are not meaningful (docs/EXPERIMENTS.md)

# The grid per predictor: attribute name on the fitted model -> values to try.
GRIDS = {
    "v1_rules": {"w_dead": [0.03, 0.05, 0.1, 0.2, 0.4], "w_team": [0.03, 0.05, 0.1, 0.2, 0.4]},
    "v2_deadball": {"w_dead": [0.05, 0.1, 0.2], "w_team": [0.05, 0.1, 0.2],
                    "w_restart": [0.5, 1.0, 2.0, 4.0]},
    "v4_gbm_nosnap": {"w_dead": [0.05, 0.1, 0.2], "w_team": [0.05, 0.1, 0.2],
                      "w_dur": [0.5, 1.0, 1.5]},
}


def pooled_spell(pairs):
    """Official pooled spellF1 of (gt, pred) pairs."""
    gt, pred = harness.pooled_arrays(pairs)
    return scoring.score_codes(gt, pred)["spell_f1_mean"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--replicates", type=int, default=1)
    args = parser.parse_args()

    predictor = predictors.get(args.name)
    grid = GRIDS[args.name]
    keys = sorted(grid)
    settings = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    inputs = harness.build_inputs(replicates=args.replicates)
    default = {key: getattr(predictor.model_for(inputs[0][1]["game"]), key) for key in keys}
    if default not in settings:
        settings.append(default)

    # decode every (setting, input) once
    results = {}                                   # (setting index, game) -> list of (gt, pred)
    for track, info, gt in inputs:
        feats = features.compute(track, info)
        model = predictor.model_for(info["game"])
        for i, setting in enumerate(settings):
            for key, value in setting.items():
                setattr(model, key, value)
            results.setdefault((i, info["game"]), []).append((gt, model.predict(feats, info)))
    games = sorted({info["game"] for _, info, _ in inputs})

    in_sample = [pooled_spell([p for g in games for p in results[(i, g)]])
                 for i in range(len(settings))]
    print("IN-SAMPLE pooled spellF1 ({} settings, {} games)".format(len(settings), len(games)))
    for i in np.argsort(in_sample)[::-1][:10]:
        print("  {:.4f}  {}".format(in_sample[i], settings[i]))

    print("\nLEAVE-ONE-GAME-OUT (pick on the other games, score on this one)")
    logo_pairs = []
    for game in games:
        others = [g for g in games if g != game]
        scores = [pooled_spell([p for g in others for p in results[(i, g)]])
                  for i in range(len(settings))]
        pick = int(np.argmax(scores))
        held = pooled_spell(results[(pick, game)])
        logo_pairs.extend(results[(pick, game)])
        print("  {:<10} picked {}  -> held-out spellF1 {:.4f}".format(game, settings[pick], held))
    print("  pooled LOGO spellF1 {:.4f}".format(pooled_spell(logo_pairs)))

    best = int(np.argmax(in_sample))
    keep = settings.index(default)
    print("\ndefault {} scores {:.4f}; best {} scores {:.4f}".format(
        default, in_sample[keep], settings[best], in_sample[best]))
    if in_sample[best] - in_sample[keep] > NOISE:
        print("-> CHANGE to the best setting (beats the default by more than {})".format(NOISE))
    else:
        print("-> KEEP the default (no setting beats it by more than {})".format(NOISE))


if __name__ == "__main__":
    main()
