"""Per-frame features from tracking: ball, players, attacking direction. Metres, team level only.

Input : a tracking dict in atd.load_cache() format (our match, or a degraded external game) and
        the info dict of harness.py (pitch, fps, half, kickoff).
Output: compute() returns a dict  name -> (N,) numpy array, one value per frame.

The same code runs on our match and on the external games, so whatever a later stage learns or
tunes on the external games means the same thing on ours. Slots are TRACKLETS, not people
(CLAUDE.md): we only compute collective quantities -- counts, team means, the nearest player of
each team -- and never follow one slot through time.

Glossary of the arrays (units in brackets, NaN where a value does not exist):
    ball_present        ball located this frame                                   (bool)
    bx, by              ball position                                             [m]
    ball_speed          ball speed from finite differences over present frames    [m/s]
    ball_accel          change of ball velocity                                   [m/s^2]
    ball_out_m          how far the ball is outside the pitch, 0 inside           [m]
    gap_len             length of the ball gap this frame is in, 0 if ball seen   [frames]
    gap_since/gap_left  frames since that gap started / until it ends             [frames]
    last_bx/by, next_bx/by   where the ball was last seen / is next seen          [m]
    last_out_m, next_out_m   ball_out_m at those two positions                    [m]
    static_len          length of the run of 'ball lying still' this frame is in  [frames]
    kick_since          frames since the latest restart-like kick (NEVER if none) [frames]
    kick_x, kick_y      where that kick happened                                  [m]
    heading_out_m       ball missing: how far outside the pitch it would be HEADING_S after
                        it vanished, at its last velocity (0 inside, NaN if seen)  [m]
    n_home, n_away      tracked objects per team
    speed_mean          mean speed of all tracked players                         [m/s]
    speed_mean_1s/_3s   the same, centred rolling mean over 1 s / 3 s             [m/s]
    decel_1s            speed_mean_1s now minus one second ago (negative = slowing down)
    speed_home/away_1s  team mean speed, rolled over 1 s                          [m/s]
    cx_home ... cy_away team centroids                                            [m]
    sx_/sy_home/away    spread (std) of each team's x / y, rolled 1 s             [m]
    len_/wid_home/away  each team's extent along x / y (max - min), rolled 1 s    [m]
    vx_home/away        each team's centroid velocity along x over the last 1 s   [m/s]
    d_home, d_away      distance from the ball to the nearest player of each team [m]
    n_near5, n_near10   players (both teams) within 5 m / 10 m of the ball
    half                1 or 2 on graded frames, 0 elsewhere (copied from info)
    pre_kickoff         graded frame before its half's kick-off                   (bool)
    home_dir            +1 if home attacks towards x = pitch length, -1 if towards 0
"""

import warnings

import numpy as np

STATIC_MS = 0.5        # a ball slower than 0.5 m/s is lying still
NEAR_M = (5.0, 10.0)   # radii for "players near the ball"
KICKOFF_SPAN = 25      # frames either side of a kick-off used to see which half each team is in

# A "restart-like kick" (M2): the ball lies still >= KICK_STILL frames and then moves faster
# than KICK_MS within KICK_WINDOW frames, or it reappears after a gap of >= REAPPEAR_GAP frames
# and moves faster than REAPPEAR_MS within KICK_WINDOW frames. Restarts look like this; so do
# some moments of live play -- what such a kick MEANS is counted on the external labels.
KICK_STILL, KICK_MS, KICK_WINDOW = 10, 4.0, 8
REAPPEAR_GAP, REAPPEAR_MS = 50, 2.0
NEVER = 10_000         # "frames since" value when the event never happened
HEADING_S = 0.4        # where a vanishing ball would be after 0.4 s at its last velocity


# --- small array helpers ---------------------------------------------------------------------

def runs_of(mask):
    """(start, stop) pairs of the True runs of a boolean array, stop exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def run_position(mask):
    """For each frame of a True run: (run length, frames since run start, frames to run end).

    All three are 0 on False frames. Example: mask [0,1,1,1,0] -> total [0,3,3,3,0],
    since [0,0,1,2,0], left [0,2,1,0,0].
    """
    n = len(mask)
    total = np.zeros(n, dtype=np.int32)
    since = np.zeros(n, dtype=np.int32)
    left = np.zeros(n, dtype=np.int32)
    for start, stop in runs_of(mask):
        steps = np.arange(stop - start, dtype=np.int32)
        total[start:stop] = stop - start
        since[start:stop] = steps
        left[start:stop] = steps[::-1]
    return total, since, left


def last_index(present):
    """Index of the most recent True frame at or before each frame (-1 if none yet)."""
    idx = np.where(present, np.arange(len(present)), -1)
    return np.maximum.accumulate(idx)


def next_index(present):
    """Index of the next True frame at or after each frame (len if none left)."""
    n = len(present)
    idx = np.where(present, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def take_or_nan(values, index):
    """values[index], with NaN where index is out of range (-1 or len)."""
    out = np.full(len(index), np.nan)
    ok = (index >= 0) & (index < len(values))
    out[ok] = values[index[ok]]
    return out


def rolling_nanmean(x, window):
    """Centred rolling mean over `window` frames that ignores NaN (NaN if the window is empty)."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    half = window // 2
    idx = np.arange(len(x))
    low = np.clip(idx - half, 0, len(x))
    high = np.clip(idx + half + 1, 0, len(x))
    n = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (sums[high] - sums[low]) / np.maximum(n, 1), np.nan)


def nanmean_rows(values, mask):
    """Mean over columns where mask is True, per row; NaN for a row with no True column."""
    total = np.where(mask, values, 0.0).sum(axis=1)
    count = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def outside_distance(x, y, length, width):
    """Metres outside the pitch box (0 inside, NaN where the position is unknown)."""
    with np.errstate(invalid="ignore"):
        return np.maximum.reduce([np.zeros_like(x), -x, x - length, -y, y - width])


# --- the feature groups ---------------------------------------------------------------------

def ball_velocity(bx, by, fps):
    """Ball velocity (m/s) per frame from finite differences, only across frames where it is seen.

    Prefers a centred difference over +/-2 frames (smoother), falls back to +/-1, then to a
    one-sided step. NaN where no neighbour is seen.
    """
    n = len(bx)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    for back, ahead in ((2, 2), (1, 1), (1, 0), (0, 1)):
        lo = np.arange(n) - back
        hi = np.arange(n) + ahead
        ok = (lo >= 0) & (hi < n) & np.isnan(vx)
        ok[ok] &= ~np.isnan(bx[lo[ok]]) & ~np.isnan(bx[hi[ok]])
        dt = (back + ahead) / fps
        vx[ok] = (bx[hi[ok]] - bx[lo[ok]]) / dt
        vy[ok] = (by[hi[ok]] - by[lo[ok]]) / dt
    return vx, vy


def restart_kicks(present, speed):
    """Frames of restart-like kicks (see KICK_* above), sorted, each frame at most once."""
    events = set()
    with np.errstate(invalid="ignore"):
        still = present & (speed < STATIC_MS)
        for runs, min_len, min_speed in ((runs_of(still), KICK_STILL, KICK_MS),
                                         (runs_of(~present), REAPPEAR_GAP, REAPPEAR_MS)):
            for start, stop in runs:
                if stop - start < min_len:
                    continue
                hit = np.flatnonzero(speed[stop:stop + KICK_WINDOW] >= min_speed)
                if len(hit):
                    events.add(stop + int(hit[0]))
    return np.array(sorted(events), dtype=np.int64)


def since_event(events, n):
    """(frames since the latest event at or before each frame, index of that event or -1)."""
    marks = np.zeros(n, dtype=bool)
    marks[events] = True
    last = last_index(marks)
    since = np.where(last >= 0, np.arange(n) - last, NEVER)
    return since.astype(np.int32), last


def ball_features(track, info):
    """Everything about the ball; see the module glossary."""
    length, width = info["pitch"]
    fps = info["fps"]
    bx = track["ball"][:, 0] * length
    by = track["ball"][:, 1] * width
    present = ~np.isnan(bx)

    vx, vy = ball_velocity(bx, by, fps)
    speed = np.hypot(vx, vy)
    # acceleration = change of the velocity vector over +/-1 frame
    ax = np.full(len(bx), np.nan)
    ay = np.full(len(bx), np.nan)
    ax[1:-1] = (vx[2:] - vx[:-2]) * fps / 2
    ay[1:-1] = (vy[2:] - vy[:-2]) * fps / 2

    gap_len, gap_since, gap_left = run_position(~present)
    last = last_index(present)
    nxt = next_index(present)
    last_bx, last_by = take_or_nan(bx, last), take_or_nan(by, last)
    next_bx, next_by = take_or_nan(bx, nxt), take_or_nan(by, nxt)

    with np.errstate(invalid="ignore"):
        static = present & (speed < STATIC_MS)
    static_len, _, _ = run_position(static)

    # M2: restart-like kicks, and where a vanishing ball was heading
    kicks = restart_kicks(present, speed)
    kick_since, kick_at = since_event(kicks, len(bx))
    last_vx, last_vy = take_or_nan(vx, last), take_or_nan(vy, last)
    heading_out = outside_distance(last_bx + last_vx * HEADING_S, last_by + last_vy * HEADING_S,
                                   length, width)

    return {
        "kick_since": kick_since,
        "kick_x": take_or_nan(bx, kick_at), "kick_y": take_or_nan(by, kick_at),
        "heading_out_m": np.where(present, np.nan, heading_out),
        "ball_present": present,
        "bx": bx, "by": by,
        "ball_vx": vx, "ball_vy": vy,
        "ball_speed": speed,
        "ball_accel": np.hypot(ax, ay),
        "ball_out_m": outside_distance(bx, by, length, width),
        "gap_len": gap_len, "gap_since": gap_since, "gap_left": gap_left,
        "last_bx": last_bx, "last_by": last_by, "next_bx": next_bx, "next_by": next_by,
        "last_out_m": outside_distance(last_bx, last_by, length, width),
        "next_out_m": outside_distance(next_bx, next_by, length, width),
        "static_len": static_len,
    }


def player_features(track, info, bx, by):
    """Counts, speeds, centroids and distances to the ball, per team (0 home, 1 away)."""
    length, width = info["pitch"]
    fps = info["fps"]
    px = track["px"] * length
    py = track["py"] * width
    seen = ~np.isnan(px)
    home = track["team"] == 0
    away = ~home
    speed = np.where(seen, np.nan_to_num(track["ps"].astype(np.float64)), 0.0)

    speed_mean = nanmean_rows(speed, seen)
    speed_1s = rolling_nanmean(speed_mean, fps)
    decel = np.full(len(px), np.nan)
    decel[fps:] = speed_1s[fps:] - speed_1s[:-fps]

    out = {
        "n_home": seen[:, home].sum(axis=1).astype(np.int16),
        "n_away": seen[:, away].sum(axis=1).astype(np.int16),
        "speed_mean": speed_mean,
        "speed_mean_1s": speed_1s,
        "speed_mean_3s": rolling_nanmean(speed_mean, 3 * fps),
        "decel_1s": decel,
        "speed_home_1s": rolling_nanmean(nanmean_rows(speed[:, home], seen[:, home]), fps),
        "speed_away_1s": rolling_nanmean(nanmean_rows(speed[:, away], seen[:, away]), fps),
    }
    for name, team in (("home", home), ("away", away)):
        out["cx_" + name] = nanmean_rows(np.nan_to_num(px[:, team]), seen[:, team])
        out["cy_" + name] = nanmean_rows(np.nan_to_num(py[:, team]), seen[:, team])

    # Team shape (M3): spread, extent, and how the team moves, rolled over 1 s. These carry
    # possession information when the ball is not seen: the team in possession spreads out
    # and pushes up, the defending team compacts.
    for name, team in (("home", home), ("away", away)):
        x = np.where(seen[:, team], px[:, team], np.nan)
        y = np.where(seen[:, team], py[:, team], np.nan)
        out["sx_" + name] = rolling_nanmean(np.nanstd(x, axis=1), fps)
        out["sy_" + name] = rolling_nanmean(np.nanstd(y, axis=1), fps)
        out["len_" + name] = rolling_nanmean(np.nanmax(x, axis=1) - np.nanmin(x, axis=1), fps)
        out["wid_" + name] = rolling_nanmean(np.nanmax(y, axis=1) - np.nanmin(y, axis=1), fps)
        cx = out["cx_" + name]
        vx = np.full(len(px), np.nan)
        vx[fps:] = (cx[fps:] - cx[:-fps])                      # metres per second along x
        out["vx_" + name] = vx

    # Distances to the ball: NaN where slot or ball is missing; nearest per team.
    with np.errstate(invalid="ignore"):
        dist = np.hypot(px - bx[:, None], py - by[:, None])
        dist = np.where(np.isnan(dist), np.inf, dist)
        for name, team in (("home", home), ("away", away)):
            nearest = dist[:, team].min(axis=1) if team.any() else np.full(len(px), np.inf)
            out["d_" + name] = np.where(np.isfinite(nearest), nearest, np.nan)
        out["n_near5"] = (dist < NEAR_M[0]).sum(axis=1).astype(np.int16)
        out["n_near10"] = (dist < NEAR_M[1]).sum(axis=1).astype(np.int16)
    return out


def home_attack_direction(cx_home, cx_away, half, kickoff):
    """+1 where home attacks towards x = pitch length, -1 towards x = 0, 0 outside the halves.

    At a kick-off every player must be in his own half, so the team whose centroid is nearer
    x = 0 around the kick-off frame defends that goal and attacks towards the far one. Measured
    for each half separately (teams swap ends); nothing is hard-coded.
    """
    out = np.zeros(len(half), dtype=np.int8)
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        k = int(kickoff.get(number, rows[0]))
        window = np.arange(max(rows[0], k - KICKOFF_SPAN), min(rows[-1] + 1, k + KICKOFF_SPAN))
        gap = np.nanmedian(cx_away[window] - cx_home[window]) if len(window) else np.nan
        if np.isnan(gap):                    # fall back: whole-half average positions
            gap = np.nanmean(cx_away[rows] - cx_home[rows])
        out[rows] = 1 if gap > 0 else -1
    return out


def compute(track, info):
    """All features for one recording, as a dict of (N,) arrays. See the module glossary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN medians are handled
        feats = ball_features(track, info)
        feats.update(player_features(track, info, feats["bx"], feats["by"]))
        half = np.asarray(info["half"]).astype(np.int8)
        frame = np.arange(len(half))
        pre = np.zeros(len(half), dtype=bool)
        for number, k in info["kickoff"].items():
            pre |= (half == number) & (frame < k)
        feats["half"] = half
        feats["pre_kickoff"] = pre
        feats["pitch"] = tuple(info["pitch"])     # scalars that travel with the arrays
        feats["fps"] = int(info["fps"])
        feats["home_dir"] = home_attack_direction(feats["cx_home"], feats["cx_away"], half,
                                                  info["kickoff"])
    return feats
