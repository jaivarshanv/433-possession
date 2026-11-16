"""Does a predictor under-segment? Possession counts, stoppages, merges vs splits. Aggregates only.

    python scripts/14_possession_diagnostic.py baseline_smooth50 v2_deadball v5_gbm_centred

Per external game (degradation replicate 0) and pooled:
  possessions   predicted vs true count and mean duration
  dead ball     predicted vs true D share and stoppage count
  merges        true label changes we missed   (official transition matcher, any type, +/- 1 s)
  splits        predicted label changes with no true one within +/- 1 s
A bias shows as merges >> splits and too few possessions on EVERY game.
Then, on our match (no labels): possessions per graded minute and the D share.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import config, decoder, frames, harness, predictors, scoring  # noqa: E402

TOL = 25


def counts(gt, pred):
    """Aggregate segmentation numbers of one (gt, pred) pair."""
    pm = scoring._pm()
    runs = pm.valid_runs(gt)
    out = {}
    graded = gt != config.X
    for name, codes in (("true", gt), ("pred", pred)):
        st = pm.possession_stats(codes, runs)
        out[name] = (st["n_spells"], st["mean_duration_s"], float(np.mean(codes[graded] == config.D)),
                     sum(1 for s, _, _ in decoder.segments_of(np.where(graded, codes, config.X))
                         if s == config.D))
    _, missed, spurious = pm.match_transitions(pm.transitions(gt, runs),
                                               pm.transitions(pred, runs), TOL, "none")
    out["merges"], out["splits"] = len(missed), len(spurious)
    out["spellF1"] = pm.spell_scores(gt, pred, runs)["mean_f1"]
    return out


def line(label, c):
    return ("  {:<10} poss {:>4} / true {:>4} ({:4.1f} s / {:4.1f} s) | D share {:.3f} / {:.3f} | "
            "stoppages {:>3} / {:>3} | merges {:>4} splits {:>4} | spellF1 {:.3f}").format(
        label, c["pred"][0], c["true"][0], c["pred"][1], c["true"][1], c["pred"][2],
        c["true"][2], c["pred"][3], c["true"][3], c["merges"], c["splits"], c["spellF1"])


def main(names):
    inputs = harness.build_inputs(replicates=1)
    track, info = harness.target_inputs()
    graded = frames.graded_mask()
    minutes = graded.sum() / config.FPS / 60
    for name in names:
        predict = predictors.get(name)
        print(name)
        pairs = []
        for trk, inf, gt in inputs:
            pred = predict(trk, inf)
            pairs.append((gt, pred))
            print(line(inf["game"], counts(gt, pred)))
        print(line("POOLED", counts(*harness.pooled_arrays(pairs))))
        ours = predict(track, info)
        n = frames.describe_possessions(ours)["n_possessions"]
        print("  {:<10} {} possessions = {:.2f} per graded minute | D share {:.3f}".format(
            "our match", n, n / minutes, float(np.mean(ours[graded] == config.D))))


if __name__ == "__main__":
    main(sys.argv[1:] or ["baseline_smooth50", "v2_deadball", "v4_gbm_nosnap", "v5_gbm_centred"])
