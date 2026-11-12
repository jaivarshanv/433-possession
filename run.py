"""Team 433 -- regenerate our submission from the data bundle, end to end.

    python run.py                  # -> submissions/433.csv, validated with the official checks
    python run.py --check-repro    # rebuild twice from EMPTY caches, compare sha256 values

What it does, all automatically (nothing is labelled or corrected by hand):
  1. checks the match bundle is in data/ (or wherever MTAH_DATA_DIR points)
  2. parses the tactical tracking into cache/ (once; MTAH_CACHE_DIR moves the cache)
  3. downloads Metrica's open sample games (pinned commit, sha256-checked) into external/,
     derives their labels from their event data, degrades their tracking like ours
  4. fits the chosen predictor (src/possession/predictors.py: CHOSEN) on those EXTERNAL games
     only -- nothing is fitted on our match
  5. predicts every frame of our match, writes submissions/433.csv, validates it

Prints aggregates only (label shares, possession count, sha256): never frame-level output.
Deterministic: fixed seeds, pinned dependencies (requirements.txt), sorted iteration.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from possession import atd, config, frames, harness, metrica, predictors, submission  # noqa: E402

TEAM = "433"


def build(out_path):
    """Steps 1-5. Returns the written path."""
    start = time.time()
    missing = [str(config.asset(name)) for name in ("atd_raw", "atd_meta", "smart_tagging")
               if not config.asset(name).exists()]
    if missing:
        sys.exit("match data not found:\n  " + "\n  ".join(missing)
                 + "\nUnpack the bundle into data/ (README) or set MTAH_DATA_DIR.")
    atd.build_cache("tactical")                                    # 2 (skipped if present)
    metrica.download(verbose=True)                                 # 3 (skipped if present)
    track, info = harness.target_inputs()
    codes = harness.check_prediction(predictors.get(predictors.CHOSEN)(track, info),
                                     config.N_FRAMES, config.GAME)  # 4 + 5
    path = submission.write_submission(codes, TEAM, out_path)      # validates, or raises
    graded = frames.graded_mask()
    shares = {config.LABELS[k]: round(float((codes[graded] == k).mean()), 3) for k in range(4)}
    print("predictor  {}".format(predictors.CHOSEN))
    print("written    {}  (official validation passed)".format(path))
    print("sha256     {}".format(submission.sha256_of(path)))
    print("graded label shares {}  possessions {}".format(shares, frames.describe_possessions(codes)))
    print("runtime    {:.0f} s".format(time.time() - start))
    return path


def check_repro():
    """Rebuild twice, each time in a brand-new empty cache, and compare the files byte for byte."""
    main_path = config.SUBMISSIONS_DIR / "{}.csv".format(TEAM)
    hashes = []
    for run in (1, 2):
        env = dict(os.environ)
        env["MTAH_CACHE_DIR"] = str(config.ROOT / "cache" / "repro_run{}".format(run))
        env["MTAH_EXTERNAL_CACHE_DIR"] = str(config.EXTERNAL_DIR / "derived_repro_run{}".format(run))
        for folder in (env["MTAH_CACHE_DIR"], env["MTAH_EXTERNAL_CACHE_DIR"]):
            shutil.rmtree(folder, ignore_errors=True)               # start from empty
        out = config.SUBMISSIONS_DIR / "repro_run{}.csv".format(run)
        print("--- run {} from empty caches".format(run), flush=True)
        subprocess.run([sys.executable, "-I", str(Path(__file__)), "--out", str(out)],
                       env=env, check=True)
        hashes.append(submission.sha256_of(out))
        for folder in (env["MTAH_CACHE_DIR"], env["MTAH_EXTERNAL_CACHE_DIR"]):
            shutil.rmtree(folder, ignore_errors=True)
        out.unlink()
    print("--- sha256 run 1 {}\n    sha256 run 2 {}".format(*hashes))
    if main_path.exists():
        print("    sha256 433   {}".format(submission.sha256_of(main_path)))
        hashes.append(submission.sha256_of(main_path))
    same = len(set(hashes)) == 1
    print("REPRODUCIBLE: identical files." if same else "NOT REPRODUCIBLE: files differ.")
    return same


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="output CSV (default submissions/433.csv)")
    parser.add_argument("--check-repro", action="store_true",
                        help="rebuild twice from empty caches and compare sha256")
    args = parser.parse_args()
    if args.check_repro:
        sys.exit(0 if check_repro() else 1)
    build(args.out)


if __name__ == "__main__":
    main()
