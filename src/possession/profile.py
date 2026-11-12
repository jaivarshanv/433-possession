"""Aggregate "what does this tracking look like" statistics, identical for any recording.

Used to calibrate degrade.py: we measure our match's automatic tracking and a degraded external
game with the SAME function and compare the numbers side by side (scripts/11_profile_tracking.py).
Everything returned is an aggregate (shares, percentiles, counts) -- safe to print, and it
contains nothing frame-level about the match.

Input is a tracking dict in the cache format of atd.py (ball (N,3), px, py (N,S), ps (N,S),
team (S,)) plus a boolean mask of the frames to profile (graded frames) and the pitch size.
"""

import warnings

import numpy as np

from . import config

SPEED_BINS = (0.0, 1.0, 1.5, 2.0, 2.5, np.inf)        # m/s, mean speed of tracked players
CROWD_RADIUS_M = 3.0                                   # "players near the ball"
BALL_SPEED_BINS = (0.0, 5.0, 15.0, np.inf)             # m/s between consecutive frames
JUMP_M = 1.5                                           # ball step > 1.5 m in 1 frame = 37.5 m/s


def run_lengths(mask):
    """Lengths of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def pct(values, qs=(10, 50, 90, 99)):
    """Rounded percentiles, or None for an empty array."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return [round(float(v), 2) for v in np.percentile(values, qs)]


def mean_player_speed(track):
    """Per frame: mean speed (m/s) of the tracked objects. NaN on a frame with nobody."""
    speed = np.where(np.isnan(track["px"]), np.nan, track["ps"].astype(np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # "mean of empty slice" is fine here
        return np.nanmean(speed, axis=1)


def ball_profile(track, mask, pitch):
    """Ball statistics: how much is missing, in what bursts, and where gaps start."""
    length, width = pitch
    bx, by = track["ball"][:, 0], track["ball"][:, 1]
    missing = np.isnan(bx)
    out = {"missing_share": round(float(missing[mask].mean()), 4)}

    # Gap and visible-run lengths (frames), counted inside the profiled frames only.
    gaps = run_lengths(missing & mask)
    seen = run_lengths(~missing & mask)
    minutes = mask.sum() / config.FPS / 60
    out["gaps_per_min"] = round(len(gaps) / minutes, 2)
    out["gap_len_p10_50_90_99"] = pct(gaps)
    out["gap_len_mean"] = round(float(gaps.mean()), 1) if len(gaps) else None
    out["share_of_missing_in_gaps_over_2s"] = round(
        float(gaps[gaps > 2 * config.FPS].sum() / max(gaps.sum(), 1)), 3)
    out["visible_len_p10_50_90"] = pct(seen, (10, 50, 90))

    with np.errstate(invalid="ignore"):
        outside = (bx < 0) | (bx > 1) | (by < 0) | (by > 1)
    out["outside_pitch_share"] = round(float((outside & mask).sum() / mask.sum()), 4)

    # Step size between consecutive visible frames: ball speed, and implausible jumps.
    step = np.hypot(np.diff(bx) * length, np.diff(by) * width)
    both = ~missing[1:] & ~missing[:-1] & mask[1:]
    out["jump_share"] = round(float((step[both] > JUMP_M).mean()), 4)

    # Where do gaps START? Hazard = P(missing at t+1 | visible at t), split by context at t.
    start = np.zeros(len(bx), dtype=bool)
    start[:-1] = ~missing[:-1] & missing[1:]
    candidate = ~missing & mask
    candidate[-1] = False
    dist = np.hypot((track["px"] - bx[:, None]) * length, (track["py"] - by[:, None]) * width)
    with np.errstate(invalid="ignore"):
        near = np.sum(dist < CROWD_RADIUS_M, axis=1)       # NaN compares False -> not counted
    hazard = {}
    for k, label in ((0, "0"), (1, "1"), (2, "2"), (3, "3+")):
        sel = candidate & ((near >= 3) if k == 3 else (near == k))
        hazard[label] = round(float(start[sel].mean()), 4) if sel.any() else None
    out["gap_start_hazard_by_players_within_3m"] = hazard

    # Ball speed AT t is the step from t-1 to t (the step after t is unknown: t+1 is missing).
    speed = np.full(len(bx), np.nan)
    speed[1:] = step * config.FPS
    hazard = {}
    with np.errstate(invalid="ignore"):
        for low, high in zip(BALL_SPEED_BINS[:-1], BALL_SPEED_BINS[1:]):
            sel = candidate & (speed >= low) & (speed < high)
            hazard["{:g}-{:g}".format(low, high)] = \
                round(float(start[sel].mean()), 4) if sel.any() else None
    out["gap_start_hazard_by_ball_speed"] = hazard

    # Missing share by collective player speed: a label-free proxy for "is play stopped".
    team_speed = mean_player_speed(track)
    by_speed = {}
    with np.errstate(invalid="ignore"):
        for low, high in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
            sel = mask & (team_speed >= low) & (team_speed < high)
            by_speed["{:g}-{:g}".format(low, high)] = (
                round(float(missing[sel].mean()), 3) if sel.any() else None,
                round(float(sel.sum() / mask.sum()), 3))
    out["missing_share_by_mean_player_speed (missing, share of frames)"] = by_speed

    # Gap lengths split by how fast the players were moving when the gap began: gaps that start
    # in fast play are tracking losses, gaps that start with everyone slow are mostly stoppages.
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    run_start = np.flatnonzero(edges == 1)
    run_len = np.flatnonzero(edges == -1) - run_start
    gap_starts = np.flatnonzero(start & mask)               # visible at t, missing from t+1
    lengths = run_len[np.searchsorted(run_start, gap_starts + 1)]
    speed_at = team_speed[gap_starts]
    with np.errstate(invalid="ignore"):
        for label, sel in (("players_fast(>=2m/s)", speed_at >= 2.0),
                           ("players_slow(<1.2m/s)", speed_at < 1.2)):
            out["gap_len_p10_50_90 when " + label] = pct(lengths[sel], (10, 50, 90))
            out["gaps_per_min when " + label] = round(int(sel.sum()) / minutes, 2)

    # How does the ball REAPPEAR after a gap of >= 10 frames? Lying still (a restart about to
    # be taken, as in corrected data) or already moving (picked up again after the kick)?
    ends = np.flatnonzero(edges == -1)                      # first visible frame after a gap
    long_gap = (run_len >= 10) & (ends + 3 < len(bx))
    ends = ends[long_gap & mask[np.minimum(ends, len(bx) - 1)]]
    after = np.hypot((bx[ends + 3] - bx[ends]) * length, (by[ends + 3] - by[ends]) * width)
    after = after[np.isfinite(after)] * config.FPS / 3
    out["ball_speed_ms_3f_after_reappearing_p25_50_75"] = pct(after, (25, 50, 75))
    out["share_reappearing_moving(>3m/s)"] = round(float((after > 3).mean()), 3) if len(after) else None
    return out


def player_profile(track, mask, pitch):
    """Player statistics: objects per frame, tracklet lifetimes, jitter, speed field."""
    length, width = pitch
    present = ~np.isnan(track["px"])
    count = present.sum(axis=1)[mask]
    out = {
        "objects_mean": round(float(count.mean()), 2),
        "objects_p1_5_50_95_99": pct(count, (1, 5, 50, 95, 99)),
        "share_over_22": round(float((count > 22).mean()), 4),
        "per_team_mean": [round(float(present[mask][:, track["team"] == t].sum(1).mean()), 2)
                          for t in (0, 1)],
        "n_slots": int(present.shape[1]),
    }
    bins = ((0, 10), (11, 15), (16, 19), (20, 21), (22, 22), (23, 99))
    out["share_of_frames_with_count"] = {
        "{}-{}".format(low, high): round(float(((count >= low) & (count <= high)).mean()), 4)
        for low, high in bins}
    # Episodes where many objects vanish at once (count < 18): how often, how long.
    low_count = np.zeros(len(mask), dtype=bool)
    low_count[mask] = count < 18
    episodes = run_lengths(low_count)
    minutes = mask.sum() / config.FPS / 60
    out["low_count(<18)_episodes_per_min"] = round(len(episodes) / minutes, 2)
    out["low_count(<18)_episode_len_p10_50_90"] = pct(episodes, (10, 50, 90))
    ball_missing = np.isnan(track["ball"][:, 0])[mask]
    out["objects_mean when ball missing / visible"] = [
        round(float(count[ball_missing].mean()), 2), round(float(count[~ball_missing].mean()), 2)]

    # Tracklets = runs of a slot being filled. Count the runs that start inside the mask.
    lengths, starts = [], 0
    for slot in range(present.shape[1]):
        edges = np.diff(np.concatenate(([0], present[:, slot].astype(np.int8), [0])))
        s, e = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        keep = mask[np.minimum(s, len(mask) - 1)]
        lengths.extend((e - s)[keep].tolist())
        starts += int(keep.sum())
    out["tracklet_starts_per_min"] = round(starts / minutes, 2)
    out["tracklet_len_s_p10_50_90"] = pct(np.array(lengths) / config.FPS, (10, 50, 90))

    # Jitter: |second difference| of position within a slot, in CENTIMETRES per frame^2.
    x, y = track["px"] * length * 100, track["py"] * width * 100
    acc = np.hypot(x[2:] - 2 * x[1:-1] + x[:-2], y[2:] - 2 * y[1:-1] + y[:-2])[mask[1:-1]]
    acc = acc[np.isfinite(acc)]
    out["accel_cm_per_frame2_p50_90_99_999"] = pct(acc, (50, 90, 99, 99.9))

    speed = track["ps"][mask][present[mask]]
    out["speed_ms_p50_95_999"] = pct(speed[np.isfinite(speed)], (50, 95, 99.9))
    return out


def tracking_profile(track, mask, pitch):
    """Both profiles in one dict."""
    return {"ball": ball_profile(track, mask, pitch), "players": player_profile(track, mask, pitch)}


def flatten(profile, prefix=""):
    """Nested profile dict -> flat {"ball.missing_share": value} for side-by-side printing."""
    flat = {}
    for key, value in profile.items():
        name = prefix + key
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = value
    return flat
