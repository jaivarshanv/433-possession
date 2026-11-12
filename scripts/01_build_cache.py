"""Step 1: parse the ATD once into cache/*.npz, and report its health in aggregate.

    python scripts/01_build_cache.py            # tactical (+ broadcast if present)
    python scripts/01_build_cache.py --force    # rebuild even if the cache exists

Output is aggregates only: line counts, coverage percentages, object counts, speed percentiles.
These numbers should match DATA.md (ball on 56.9% of all frames / 59.3% of graded frames,
about 20.5 tracked objects per graded frame). If they don't, the parser is wrong.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd, config, frames  # noqa: E402


def pct(mask):
    """Share of True values as a percentage string."""
    return "{:.1f}%".format(100.0 * np.mean(mask)) if len(mask) else "n/a"


def report(source, data, stats):
    """Print coverage statistics for one recording."""
    n = len(data["seen"])
    ball_ok = np.isfinite(data["ball"][:, 0]) & np.isfinite(data["ball"][:, 1])
    present = np.isfinite(data["px"]) & np.isfinite(data["py"])
    n_objects = present.sum(axis=1)
    team = data["team"]

    print("\n== {}  ({} frames, {} slots)".format(source, n, len(team)))
    print("  home = {!r}, away = {!r}".format(data["home_name"], data["away_name"]))
    if stats:
        print("  parse: {} lines, malformed {}, out of range {}, duplicates {}".format(
            stats.get("lines", 0), stats.get("malformed_lines", 0),
            stats.get("frame_out_of_range", 0), stats.get("duplicate_frames", 0)))
        print("  slot groups per line: {}".format(stats["slot_group_counts"]))
        print("  numbers in ball group: {}".format(stats["ball_field_counts"]))
    print("  frames with a line: {}".format(pct(data["seen"])))
    print("  ball located, all frames: {}".format(pct(ball_ok)))

    if source == "tactical":
        graded = frames.graded_mask(n)
        print("  ball located, graded frames: {}   (DATA.md: 59.3%)".format(pct(ball_ok[graded])))
        g_objects = n_objects[graded]
        print("  tracked objects per graded frame: mean {:.1f}, p5 {:.0f}, p95 {:.0f}, "
              "frames with >22: {}   (DATA.md: mean 20.5)".format(
                  g_objects.mean(), np.percentile(g_objects, 5), np.percentile(g_objects, 95),
                  pct(g_objects > 22)))
        home_n = present[graded][:, team == 0].sum(axis=1)
        away_n = present[graded][:, team == 1].sum(axis=1)
        print("  per team per graded frame: home mean {:.1f}, away mean {:.1f}".format(
            home_n.mean(), away_n.mean()))
        out = ball_ok[graded] & ((data["ball"][graded, 0] < 0) | (data["ball"][graded, 0] > 1) |
                                 (data["ball"][graded, 1] < 0) | (data["ball"][graded, 1] > 1))
        print("  located ball outside the 0..1 pitch box (graded): {}".format(pct(out)))

    third = data["ball"][:, 2]
    finite = np.isfinite(third)
    if finite.any():
        print("  3rd ball field: finite on {}, p5 {:.2f}, median {:.2f}, p95 {:.2f}  "
              "(height in m? speed? -> decide from these numbers)".format(
                  pct(finite), np.percentile(third[finite], 5), np.median(third[finite]),
                  np.percentile(third[finite], 95)))
    speeds = data["ps"][np.isfinite(data["ps"])]
    if len(speeds):
        print("  player speed m/s: p50 {:.2f}, p95 {:.2f}, p99.9 {:.2f}".format(
            np.percentile(speeds, 50), np.percentile(speeds, 95), np.percentile(speeds, 99.9)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = parser.parse_args()

    for source in ("tactical", "broadcast"):
        if not config.asset("atd_raw", source).exists():
            print("\n{}: no ATD file, skipped".format(source))
            continue
        start = time.time()
        path, stats = atd.build_cache(source, force=args.force)
        print("\n{}: cache {} ({:.0f} s)".format(
            source, "built" if stats else "already present", time.time() - start))
        report(source, atd.load_cache(source), stats)

    print("\nNext: python scripts/02_reproduce_baseline.py")


if __name__ == "__main__":
    main()
