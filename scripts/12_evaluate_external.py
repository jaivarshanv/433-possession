"""Stage 2: score predictors on the degraded external games with the official metric.

    python scripts/12_evaluate_external.py baseline_smooth50
    python scripts/12_evaluate_external.py baseline_raw baseline_smooth25 baseline_smooth50 \
        --replicates 3 --bootstrap 1000 --public-check

Predictor names come from src/possession/predictors.py (REGISTRY). For each one it prints the
official scores per external game and POOLED over all of them (the number that decides), then
a ready-to-paste row for docs/EXPERIMENTS.md.

  --replicates R   degrade every game R times with different seeds (averages out the noise of
                   the simulated tracker); pooled over all of them
  --bootstrap N    official bootstrap 5-95% interval on the pooled spellF1 (replicate 0 only,
                   so the possessions resampled are really distinct)
  --clean          give the predictor the CLEAN tracking: diagnostic only, never for selection
  --public-check   also run on our match and score the public 5 minutes -- a PIPELINE CHECK
                   (format, alignment), never a selection criterion. Prints aggregates only.
"""

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import config, frames, harness, predictors, scoring  # noqa: E402


def line(name, result):
    """One aligned line of the numbers that matter for one evaluate() result."""
    s = scoring.summarize(result)
    return ("{:<14} spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  alive/dead {alive_dead_F1:.4f}  "
            "F1 H/A/D {F1_H:.3f}/{F1_A:.3f}/{F1_D:.3f}  poss pred {pred_possessions} "
            "({pred_mean_s} s)  gt {gt_possessions} ({gt_mean_s} s)").format(name, **s)


def public_check(predict):
    """Run a predictor on OUR match; return (public-window result, full-match possession stats).

    Only aggregates leave this function.
    """
    track, info = harness.target_inputs()
    pred = harness.check_prediction(predict(track, info), config.N_FRAMES, config.GAME)
    return scoring.score_codes(scoring.load_public_gt_codes(), pred), \
        frames.describe_possessions(pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=["baseline_smooth50"])
    parser.add_argument("--games", type=int, nargs="+", default=list(config.METRICA_GAMES))
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--public-check", action="store_true")
    args = parser.parse_args()

    inputs = harness.build_inputs(args.games, args.replicates, clean=args.clean)
    mode = "CLEAN tracking (diagnostic)" if args.clean else \
        "degraded x{}".format(args.replicates)
    print("external games {} -- {}\n".format(args.games, mode))

    rows = []
    for name in args.names:
        out = harness.evaluate(predictors.get(name), inputs, bootstrap_n=args.bootstrap)
        print(name)
        for game, result in out["per_game"].items():
            print("  " + line(game, result))
        pooled = out["pooled"]
        print("  " + line("POOLED", pooled))
        print("  {:<14} {}".format("stoppages", out["diagnostics"]))
        interval = ""
        if out["interval"]:
            interval = " [{:.3f}-{:.3f}]".format(*out["interval"])
            print("  pooled spellF1 5-95% bootstrap interval{}  (n={}, replicate 0)".format(
                interval, args.bootstrap))
        public = "—"
        if args.public_check:
            result, full = public_check(predictors.get(name))
            print("  " + line("public 5 min", result) + "   [PIPELINE CHECK ONLY]")
            print("  {:<14} full-match predicted possessions {}".format("", full))
            public = "{:.4f}".format(result["spell_f1_mean"])
        print()
        s = scoring.summarize(pooled)
        rows.append("| {} | {} | | {:.4f}{} | {:.4f} | {} | {} / {} s | games {}, {}; "
                    "gt {} poss / {} s |".format(
                        datetime.date.today().isoformat(), name, s["spellF1"], interval,
                        s["macroF1"], public, s["pred_possessions"], s["pred_mean_s"],
                        "+".join(str(g) for g in args.games), mode,
                        s["gt_possessions"], s["gt_mean_s"]))

    print("rows for docs/EXPERIMENTS.md (fill in the 'change' column):")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
