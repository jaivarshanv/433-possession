"""Step 2: prove our parser is right by reproducing the official baseline exactly.

The organisers committed four baseline CSVs (official/evaluation/results/), produced from the
ATD and nothing else. If we recompute "nearest tracked player to the ball" from OUR cache and
apply THEIR smoothing function, we must get identical files. Any differing frame means our
parsing or alignment is off.

    python scripts/02_reproduce_baseline.py

Prints only: how many frames differ per variant, and the public-window score (pipeline check).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd, config, frames, scoring  # noqa: E402
from possession.baselines import nearest_player_team  # noqa: E402  (vectorised official loop)


def main():
    config.add_official_to_path()
    import baseline as official_baseline  # official/starter/baseline.py

    data = atd.load_cache("tactical")
    codes = nearest_player_team(data)

    gt = scoring.load_public_gt_codes()
    all_match = True
    for name, window in official_baseline.VARIANTS:
        ours = official_baseline.smooth(codes, window)
        reference_path = config.OFFICIAL_RESULTS / "{}.csv".format(name)
        reference = pd.read_csv(reference_path)["label"].astype(str).values
        n_diff = int(np.sum(ours != reference))
        all_match &= n_diff == 0
        print("{:<20} frames differing from official file: {}".format(name, n_diff))
        if name == "baseline_smooth50":
            result = scoring.score_codes(gt, scoring.labels_to_codes(ours))
            scoring.print_summary("  our reproduction of " + name, result, public=True)
            print("  full-match possessions: {}".format(
                frames.describe_possessions(scoring.labels_to_codes(ours))))

    print("\nPARSER VERIFIED: identical to the official baselines." if all_match else
          "\nMISMATCH: parser or alignment differs from the official baseline -- investigate "
          "before building anything on top.")


if __name__ == "__main__":
    main()
