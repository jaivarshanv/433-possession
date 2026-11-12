"""Validate one or more submission files and score them on the PUBLIC 5-minute window.

    python scripts/score.py submissions/v1.csv [submissions/v2.csv ...]

The public window is 17 possessions. It tells you whether the pipeline is CORRECT (frames
aligned, labels meaning what you think). It must never be used to choose between versions or
tune anything -- that is both a competition rule and statistically meaningless. Use the
external-game validation for decisions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import frames, scoring, submission  # noqa: E402


def main(paths):
    if not paths:
        print(__doc__)
        return
    gt = scoring.load_public_gt_codes()
    for path in paths:
        problem = submission.validate_file(path)
        if problem:
            print("{}: INVALID -- {}".format(path, problem))
            continue
        config_frame = scoring._pm().read_label_csv(str(path))
        pred = scoring.labels_to_codes(config_frame["label"].values)
        scoring.print_summary(Path(path).name, scoring.score_codes(gt, pred), public=True)
        print("  full-match possessions: {}".format(frames.describe_possessions(pred)))


if __name__ == "__main__":
    main(sys.argv[1:])
