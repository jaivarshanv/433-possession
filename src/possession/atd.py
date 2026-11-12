"""Parse Metrica's Automatic Tracking Data (ATD) into numpy arrays, and cache them.

The raw file is one text line per frame:

    <frame_id>:<slot 1>;<slot 2>;...;<slot 46>:<ball>

Each slot group is "x,y,speed" and the ball group is "x,y,..." (we keep up to 3 numbers).
A missing observation is written as NaN. Coordinates are normalised to 0..1 of the pitch.
frame_id counts from 1, video frames (= submission rows) count from 0, so
video_frame = frame_id - 1.

The metadata XML lists the two teams (first = home) and one <Player> per track slot with its
teamId. A slot is a TRACKLET, not a person: slots appear, vanish and swap players.

Why our own parser instead of kloppy: kloppy's to_df() on 199k frames x 46 slots is slow and
memory-hungry. Parsing once into plain arrays and caching them as .npz makes every later step
load in about a second.

Cached arrays (all indexed by video frame, length N):
    ball        (N, 3) float64  normalised ball x, y and the third ball field if present (else NaN)
    px, py      (N, S) float64  normalised slot positions, NaN when the slot is empty
    ps          (N, S) float32  slot speed in m/s
    team        (S,)   int8     0 = home slot, 1 = away slot
    seen        (N,)   bool     True if the frame had a line in the raw file
plus scalars n_ball_fields (how many numbers the ball group had) and team names.

Positions are float64 on purpose: it lets scripts/02_reproduce_baseline.py reproduce the
official baseline bit-for-bit, which is our proof the parser is right.
"""

import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from . import config

HOME, AWAY = 0, 1


def read_slot_teams(meta_path):
    """Read the metadata XML.

    Returns (team_of_slot, home_name, away_name) where team_of_slot is an int8 array with one
    entry per track slot: 0 for home, 1 for away. Same logic as the official baseline: the first
    <Team> in the file is home (FIFA/EPTS convention), slots are the <Player> entries in order.
    """
    root = ET.parse(str(meta_path)).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))

    home_id = teams[0].get("id")
    names = [team.findtext("Name") for team in teams]

    slot_team_ids = [player.get("teamId") for player in root.iter("Player")]
    if not slot_team_ids:
        raise ValueError("no <Player> entries in {}".format(meta_path))

    team_of_slot = np.array([HOME if tid == home_id else AWAY for tid in slot_team_ids],
                            dtype=np.int8)
    return team_of_slot, names[0], names[1]


def parse_group(text, width):
    """Turn one group like "0.51,0.33,5.2" into a list of `width` floats.

    Handles every way a missing value can appear: "NaN", "NaN,NaN,NaN", or an empty string.
    Extra numbers are dropped, missing ones become NaN.
    """
    values = []
    for part in text.strip().split(","):
        part = part.strip()
        values.append(float(part) if part else np.nan)  # float("NaN") is NaN
    values = values[:width]
    while len(values) < width:
        values.append(np.nan)
    return values


def parse_atd(raw_path, meta_path, n_frames):
    """Parse a whole ATD file. Returns (arrays dict, stats dict).

    The stats dict holds aggregate counts only (how many lines, how many malformed ones, the
    distribution of group counts) so a script can report on the file's health without ever
    printing its contents.
    """
    team_of_slot, home_name, away_name = read_slot_teams(meta_path)
    n_slots = len(team_of_slot)

    # Allocate everything as "missing" and fill in what the file contains.
    ball = np.full((n_frames, 3), np.nan, dtype=np.float64)
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)

    stats = Counter()
    group_counts = Counter()   # how many slot groups each line had
    ball_widths = Counter()    # how many numbers the ball group had

    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 3:
                stats["malformed_lines"] += 1
                continue

            video_frame = int(parts[0]) - 1          # frame_id counts from 1
            if not 0 <= video_frame < n_frames:
                stats["frame_out_of_range"] += 1
                continue
            if seen[video_frame]:
                stats["duplicate_frames"] += 1
            seen[video_frame] = True
            stats["lines"] += 1

            # --- players: one "x,y,speed" group per slot -------------------------------------
            groups = parts[1].split(";") if parts[1] else []
            group_counts[len(groups)] += 1
            for slot, group in enumerate(groups[:n_slots]):
                x, y, s = parse_group(group, 3)
                px[video_frame, slot] = x
                py[video_frame, slot] = y
                ps[video_frame, slot] = s

            # --- ball ---------------------------------------------------------------------
            ball_text = parts[2].strip()
            ball_widths[len(ball_text.split(","))] += 1
            ball[video_frame] = parse_group(ball_text, 3)

    arrays = {
        "ball": ball, "px": px, "py": py, "ps": ps, "seen": seen, "team": team_of_slot,
        "home_name": np.array(home_name), "away_name": np.array(away_name),
        "n_ball_fields": np.array(max(ball_widths) if ball_widths else 0),
    }
    stats = dict(stats)
    stats["n_slots"] = n_slots
    stats["slot_group_counts"] = dict(group_counts)
    stats["ball_field_counts"] = dict(ball_widths)
    return arrays, stats


def cache_path(source="tactical"):
    """Where the parsed arrays for a recording are stored."""
    return config.CACHE_DIR / "{}_{}_atd.npz".format(config.GAME, source)


def build_cache(source="tactical", force=False):
    """Parse the ATD of one recording and save it to cache/. Returns (path, stats).

    Skips the work if the cache exists, unless force=True. stats is None when skipped.
    """
    path = cache_path(source)
    if path.exists() and not force:
        return path, None
    arrays, stats = parse_atd(config.asset("atd_raw", source),
                              config.asset("atd_meta", source),
                              config.n_frames_of(source))
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path, stats


def load_cache(source="tactical"):
    """Load the parsed arrays as a plain dict. Builds the cache first if needed."""
    path, _ = build_cache(source)
    with np.load(path) as stored:
        data = {key: stored[key] for key in stored.files}
    # Unwrap 0-d arrays back into Python scalars / strings.
    for key in ("home_name", "away_name", "n_ball_fields"):
        data[key] = data[key].item()
    return data


def to_metres(x_norm, y_norm):
    """Normalised pitch coordinates (0..1) -> metres. Works on scalars or arrays."""
    return x_norm * config.PITCH_LENGTH_M, y_norm * config.PITCH_WIDTH_M
