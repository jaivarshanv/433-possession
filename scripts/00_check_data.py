"""Step 0: is the match bundle in place and does it look the way DATA.md says?

Prints AGGREGATES ONLY (file sizes, counts, percentages, stream metadata), so its output is
safe to share. It never prints rows or values from the data.

    python scripts/00_check_data.py

Checks:
  1. every expected file exists, with its size
  2. the videos: stream types, resolution, frame rate, and whether there is an AUDIO track
     (decides whether referee-whistle detection is possible at all)
  3. the ATD: line count, frame id range, contiguity, slot count, teams
  4. the homography: row count, share of valid matrices
  5. the Smart Tagging: counts per code and team, and the official time-base check
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import config  # noqa: E402
from possession.atd import read_slot_teams  # noqa: E402


def check_files():
    """Print which assets exist and their sizes. Returns True if the tactical set is complete."""
    print("== 1. files  (data dir: {})".format(config.DATA_DIR))
    tactical_ok = True
    for source, names in (("tactical", ["video", "atd_meta", "atd_raw", "homography",
                                        "smart_tagging"]),
                          ("broadcast", ["video", "atd_meta", "atd_raw", "homography"])):
        for name in names:
            path = config.asset(name, source)
            if path.exists():
                print("  ok       {:<10} {:<14} {:>9.1f} MB".format(
                    source, name, path.stat().st_size / 1e6))
            else:
                print("  MISSING  {:<10} {:<14} {}".format(source, name, path))
                if source == "tactical":
                    tactical_ok = False
    return tactical_ok


def check_video(source):
    """ffprobe the video: stream list, and above all whether audio exists."""
    path = config.asset("video", source)
    if not path.exists():
        return
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,"
             "sample_rate,channels,duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print("  {}: ffprobe failed ({}). Install ffmpeg: brew install ffmpeg".format(source, error))
        return
    streams = json.loads(out).get("streams", [])
    print("  {} video: {} stream(s)".format(source, len(streams)))
    for s in streams:
        if s.get("codec_type") == "video":
            print("    video  {} {}x{}  fps {}  frames {}  duration {} s".format(
                s.get("codec_name"), s.get("width"), s.get("height"), s.get("avg_frame_rate"),
                s.get("nb_frames"), s.get("duration")))
        elif s.get("codec_type") == "audio":
            print("    AUDIO  {}  {} Hz  {} ch  duration {} s".format(
                s.get("codec_name"), s.get("sample_rate"), s.get("channels"), s.get("duration")))
        else:
            print("    other  {} {}".format(s.get("codec_type"), s.get("codec_name")))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    print("    -> audio track present: {}".format("YES" if has_audio else "NO"))


def check_atd(source):
    """Fast structural scan of the raw ATD: only the frame id before the first ':' is read."""
    raw, meta = config.asset("atd_raw", source), config.asset("atd_meta", source)
    if not (raw.exists() and meta.exists()):
        return
    team_of_slot, home, away = read_slot_teams(meta)
    first = last = None
    count = 0
    with open(raw) as handle:
        for line in handle:
            if not line.strip():
                continue
            frame_id = int(line[:line.index(":")])
            first = frame_id if first is None else first
            last = frame_id
            count += 1
    expected = config.n_frames_of(source)
    contiguous = count > 0 and last - first + 1 == count
    print("  {} ATD: {} lines, frame_id {}..{}, contiguous {}, expected {} -> {}".format(
        source, count, first, last, contiguous, expected,
        "OK" if (count == expected and contiguous and first == 1) else "CHECK"))
    print("    slots {} (home {} / away {})   home = {!r}   away = {!r}".format(
        len(team_of_slot), int((team_of_slot == 0).sum()), int((team_of_slot == 1).sum()),
        home, away))


def check_homography(source):
    """Row count, valid share, reprojection error summary."""
    path = config.asset("homography", source)
    if not path.exists():
        return
    table = pd.read_csv(path, usecols=["frame", "reprojection_error", "valid"])
    valid = table["valid"].astype(bool)
    err = table.loc[valid, "reprojection_error"]
    print("  {} homography: {} rows, valid {:.1f}%, reprojection error median {:.2f} "
          "(p95 {:.2f})".format(source, len(table), 100 * valid.mean(), err.median(),
                                err.quantile(0.95)))


def check_tagging():
    """Counts per code / team / type and the official time-base check."""
    if not config.asset("smart_tagging").exists():
        return
    from possession.tagging import load_tagging
    tags = load_tagging()
    print("  Smart Tagging: {} instances".format(len(tags)))
    print("    codes: " + ", ".join("{} {}".format(k, v)
                                    for k, v in tags["code"].value_counts().items()))
    print("    team : " + ", ".join("{} {}".format(k, v) for k, v in
                                    tags["team"].fillna("N/A").value_counts().items()))
    print("    set-piece types: " + ", ".join("{} {}".format(k, v) for k, v in
                                              tags["type"].dropna().value_counts().items()))
    config.add_official_to_path()
    import load_smart_tagging as lst  # official module
    report = lst.check_time_base(lst.load())
    print("    time base agrees with video: {}".format(report["agrees"]))


def main():
    tactical_ok = check_files()
    print("\n== 2. video streams")
    for source in ("tactical", "broadcast"):
        check_video(source)
    if not tactical_ok:
        print("\nTactical bundle incomplete -- stopping here.")
        return
    print("\n== 3. tracking (ATD)")
    for source in ("tactical", "broadcast"):
        check_atd(source)
    print("\n== 4. homography")
    for source in ("tactical", "broadcast"):
        check_homography(source)
    print("\n== 5. smart tagging")
    check_tagging()
    print("\nDone. Next: python scripts/01_build_cache.py")


if __name__ == "__main__":
    main()
