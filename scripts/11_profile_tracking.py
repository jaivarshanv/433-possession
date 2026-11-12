"""Stage 2: does a degraded external game LOOK like our match's tracking? Aggregates only.

    python scripts/11_profile_tracking.py              # our match + every external game
    python scripts/11_profile_tracking.py --no-target  # external games only (no match data)

One column per recording, one row per statistic (profile.py defines them all):
    ours       our match, graded frames                   <- the numbers degrade.py aims for
    mN clean   Metrica sample game N as delivered (corrected tracking)
    mN degr    the same game after degrade.degrade()      <- should resemble 'ours'

Prints shares, percentiles and counts only -- nothing frame-level -- so it is safe to run with
the match data present. Labels are never used for our match.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import (atd, config, degrade, event_labels, frames, metrica,  # noqa: E402
                        profile)


def target_columns():
    """Profile of our match's tactical ATD over graded frames. No labels involved."""
    track = atd.load_cache("tactical")
    pitch = (config.PITCH_LENGTH_M, config.PITCH_WIDTH_M)
    return profile.flatten(profile.tracking_profile(track, frames.graded_mask(), pitch))


def external_columns(number, degraded):
    """Profile of one external game (clean or degraded) over its labelled frames."""
    game = metrica.load_game(number)
    labels, _ = event_labels.derive_labels(game)
    mask = labels != config.X
    track = degrade.degrade(game, labels) if degraded else degrade.clean_tracking(game)
    flat = profile.flatten(profile.tracking_profile(track, mask, game["pitch"]))
    # Only knowable on external games: how much of the missing ball falls in dead time.
    missing = np.isnan(track["ball"][:, 0])
    alive = mask & (labels != config.D)
    flat["ball.(ext only) missing_share_alive"] = round(float(missing[alive].mean()), 3)
    flat["ball.(ext only) missing_share_dead"] = round(float(missing[labels == config.D].mean()), 3)
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-target", action="store_true", help="skip our match")
    parser.add_argument("--games", type=int, nargs="+", default=list(config.METRICA_GAMES))
    parser.add_argument("--clean-only", action="store_true", help="skip degraded columns")
    args = parser.parse_args()

    columns = {}
    if not args.no_target:
        columns["ours"] = target_columns()
    for number in args.games:
        columns["m{} clean".format(number)] = external_columns(number, degraded=False)
        if not args.clean_only:
            columns["m{} degr".format(number)] = external_columns(number, degraded=True)

    keys = []
    for flat in columns.values():
        keys.extend(k for k in flat if k not in keys)
    cells = {name: [str(flat.get(key, "")) for key in keys] for name, flat in columns.items()}
    widths = {name: max(len(name), *(len(c) for c in col)) for name, col in cells.items()}
    key_width = max(len(k) for k in keys)
    print("{:<{w}}  ".format("statistic", w=key_width)
          + "  ".join("{:>{w}}".format(name, w=widths[name]) for name in columns))
    for row, key in enumerate(keys):
        print("{:<{w}}  ".format(key, w=key_width)
              + "  ".join("{:>{w}}".format(cells[name][row], w=widths[name]) for name in columns))


if __name__ == "__main__":
    main()
