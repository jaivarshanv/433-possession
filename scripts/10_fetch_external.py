"""Stage 2, step 1: download the Metrica sample games and derive their H/A/D labels.

    python scripts/10_fetch_external.py

Downloads github.com/metrica-sports/sample-data at the pinned commit into external/metrica/
(sha256-checked, skipped when already present), parses each game once into external/derived/,
derives frame-level labels from its event data (event_labels.py) and prints what they look like.
External, openly licensed data only -- nothing here touches our match.

For comparison, the public 5-minute window of our match: 17 possessions (3.4 per minute),
mean 14.2 s; the official GT has possessions as short as 0.76 s.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import config, event_labels, metrica  # noqa: E402


def describe(codes):
    """Aggregate summary of a derived label array (possessions and stoppages)."""
    config.add_official_to_path()
    import possession_metrics as pm
    runs = pm.valid_runs(codes)
    segments = pm.segments(codes, runs)
    spells = np.array([b - a for a, b, k in segments if k in (config.H, config.A)]) / config.FPS
    dead = np.array([b - a for a, b, k in segments if k == config.D]) / config.FPS
    graded = int(np.sum(codes != config.X))
    alive = int(np.sum((codes == config.H) | (codes == config.A)))
    return {
        "graded_min": round(graded / config.FPS / 60, 1),
        "possessions": len(spells),
        "per_min": round(len(spells) / (graded / config.FPS / 60), 2),
        "mean_s": round(float(spells.mean()), 1),
        "median_s": round(float(np.median(spells)), 1),
        "under_1s": round(float((spells < 1).mean()), 3),
        "dead_share": round(float(np.sum(codes == config.D) / graded), 3),
        "stoppages": len(dead),
        "stoppage_median_s": round(float(np.median(dead)), 1),
        "home_share": round(float(np.sum(codes == config.H) / alive), 3),
    }


def main():
    paths = metrica.download()
    print("{} files present and sha256-verified (commit {})\n".format(
        len(paths), config.METRICA_COMMIT[:7]))
    for number in config.METRICA_GAMES:
        game = metrica.load_game(number)
        codes, rules = event_labels.derive_labels(game)
        print("{}: {} frames, {} players, {} events".format(
            game["name"], len(codes), game["px"].shape[1], len(game["events"])))
        print("  labels  ", describe(codes))
        print("  rules   ", dict(sorted(rules.items())))


if __name__ == "__main__":
    main()
