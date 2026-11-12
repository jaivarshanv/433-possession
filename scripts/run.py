"""Predict OUR match with a registered predictor and write a validated submission file.

    python scripts/run.py --predictor v1_rules                       # -> submissions/v1_rules.csv
    python scripts/run.py --predictor v1_rules --out submissions/433.csv

The file is checked with exactly the official validation (submission.write_submission) before
this script reports success. Printed: label shares on graded frames, possession count and
durations, stoppage count, and the file's sha256 -- aggregates only, never frame-level labels
(CLAUDE.md 0.1). Learned predictors are fitted on the external games only.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import config, frames, harness, predictors, submission  # noqa: E402


def summary(codes):
    """Aggregate description of a full-match prediction (safe to print)."""
    graded = frames.graded_mask()
    shares = {config.LABELS[k]: round(float(np.mean(codes[graded] == k)), 3) for k in range(4)}
    stoppages = sum(1 for first, last in config.HALVES.values()
                    for value, _, _ in frames.runs(codes[first:last + 1]) if value == config.D)
    return shares, frames.describe_possessions(codes), stoppages


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictor", default=predictors.CHOSEN)
    parser.add_argument("--out", help="output CSV (default submissions/<predictor>.csv)")
    args = parser.parse_args()

    start = time.time()
    track, info = harness.target_inputs()
    codes = harness.check_prediction(predictors.get(args.predictor)(track, info),
                                     config.N_FRAMES, config.GAME)
    path = submission.write_submission(codes, args.predictor, args.out)   # validates, or raises
    shares, possessions, stoppages = summary(codes)

    print("predictor   {}".format(args.predictor))
    print("written     {}  (official validation passed)".format(path))
    print("sha256      {}".format(submission.sha256_of(path)))
    print("graded label shares {}".format(shares))
    print("possessions {}".format(possessions))
    print("stoppages   {}".format(stoppages))
    print("runtime     {:.0f} s".format(time.time() - start))


if __name__ == "__main__":
    main()
