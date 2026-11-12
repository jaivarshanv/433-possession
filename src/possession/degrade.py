"""Make an external game's clean tracking look like our match's automatic tracking.

Clean (corrected) tracking is far easier than ours: every player in his own column the whole
match, no duplicates, a ball that is only missing while it is out of play. A model validated on
that would be validated on the wrong problem. This module turns a clean game into a dict with
EXACTLY the keys and conventions of atd.load_cache() -- so any predictor written for our match
runs on it unchanged -- and degrades it on the way:

  BALL
    1. holes shorter than FILL_SHORT_GAP frames are interpolated (ours has none that short)
    2. after the ball leaves play it is still followed while it rolls to a stop (ours loses
       the ball when it is SLOW, corrected data drops it on the line)
    3. in some stoppages the ball is seen at the restart spot before the restart
    4. bursty losses during live play, more likely on a slow ball and in a crowd
    (the third ball field, height, stays NaN: the clean data has none)
  PLAYERS
    5. each player is lost in bursts, more often when someone is within CLOSE_M of him
    6. "view loss" episodes where a random part of the pitch's players vanish together,
       more often during stoppages
    7. identity breaks: a tracked player jumps to a new slot without any gap
    8. short-lived duplicate tracks of a real player (ours has > 22 objects on 2.4% of frames)
    9. every tracklet gets a random free slot of its team (pools of 24 / 22 like ours):
       slots are NOT people, exactly as in our ATD

Every rate below was set by running scripts/11_profile_tracking.py, which prints our match's
tracking profile next to each degraded game (aggregates only, no labels), and moving the
degraded numbers towards ours. Nothing is fitted to labels. Fixed seed: same output every run.

    clean_tracking(game)                 the clean game in cache format, for reference
    degrade(game, labels, replicate=0)   the degraded version (labels say live vs dead)
"""

import zlib

import numpy as np

from . import config

SEED = 2027
SPEED_HALF_WINDOW = 2       # speed = displacement over +/- 2 frames (0.16 s), like a tracker

# --- ball --------------------------------------------------------------------------------
FILL_SHORT_GAP = 10         # frames
EXIT_DECEL_MS2 = 6.0        # a ball that left play decelerates at 6 m/s^2 ...
EXIT_MAX_FRAMES = 40        # ... and is followed for at most 1.6 s
EXIT_HOLD_MAX = 12          # ... plus up to 0.5 s lying still before the tracker drops it
DEAD_SEEN_SHARE = 0.25      # share of stoppages where the ball is seen at the restart spot
DEAD_SEEN_MAX_FRAC = 0.5    # ... during up to this (final) fraction of the stoppage
LOSS_PER_LIVE_MIN = 0.3     # ball losses per minute of live play, before the multipliers
LOSS_SLOW_MULT = 4.0        # x4 when the ball is slower than SLOW_BALL_MS
LOSS_CROWD_MULT = 3.0       # x3 with CROWD_N or more players within CROWD_RADIUS_M
LOSS_MEDIAN, LOSS_SIGMA = 125, 1.0     # loss length in frames, lognormal (median 5 s)
SLOW_BALL_MS = 5.0
CROWD_RADIUS_M, CROWD_N = 3.0, 3

# --- players -----------------------------------------------------------------------------
SLOTS_PER_TEAM = (24, 22)   # our ATD: 24 home slots, 22 away slots
PLAYER_LOSS_PER_MIN = 0.1   # per player, before the multiplier
PLAYER_LOSS_MEDIAN, PLAYER_LOSS_SIGMA = 100, 1.3
CLOSE_M, CLOSE_MULT = 2.0, 5.0         # someone within 2 m: losses x5
SWITCH_PER_MIN = 0.03       # identity breaks per player-minute ...
SWITCH_CLOSE_MULT = 25.0    # ... x25 with someone within CLOSE_M (trackers swap in crowds)
GROUP_PER_MIN = 0.2         # view-loss episodes per minute of live play ...
GROUP_DEAD_MULT = 8.0       # ... x8 during stoppages
GROUP_MEDIAN, GROUP_SIGMA = 67, 2.0    # episode length in frames, lognormal
GROUP_FRACTION = (0.1, 0.6)            # share of players an episode hides, uniform
MAX_BURST = 1500            # no loss or episode lasts longer than 60 s
DUP_PER_MIN = 0.5           # duplicate tracks started per minute
DUP_MEDIAN, DUP_SIGMA = 40, 1.0        # duplicate lifetime in frames, lognormal
DUP_OFFSET_M = (0.3, 1.5)              # duplicate sits this far from the real player

H, A, D = config.H, config.A, config.D


# --- small helpers -----------------------------------------------------------------------

def per_frame(rate_per_minute):
    """Events per minute -> probability per frame at 25 fps."""
    return rate_per_minute / (60.0 * config.FPS)


def runs_of(mask):
    """(start, stop) pairs of the True runs of a boolean array, stop exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def lognormal_frames(rng, median, sigma, size):
    """Integer durations from a lognormal with the given median, clipped to 1..MAX_BURST."""
    lengths = np.round(median * np.exp(sigma * rng.standard_normal(size)))
    return np.clip(lengths, 1, MAX_BURST).astype(int)


def bursts(rng, hazard, median, sigma):
    """Boolean mask of frames inside randomly placed bursts.

    A burst starts on frame t with probability hazard[t] and lasts a lognormal number of frames.
    A start that falls inside a running burst is ignored, so bursts never overlap. All random
    numbers are drawn up front, so the result depends on the seed only.
    """
    n = len(hazard)
    starts = np.flatnonzero(rng.random(n) < hazard)
    lengths = lognormal_frames(rng, median, sigma, len(starts))
    lost = np.zeros(n, dtype=bool)
    end = -1
    for start, length in zip(starts.tolist(), lengths.tolist()):
        if start < end:
            continue
        end = start + length
        lost[start:end] = True
    return lost


def speed_from_positions(px, py, pitch, half=SPEED_HALF_WINDOW):
    """Per-column speed in m/s from normalised positions, central difference over +/- half frames.

    NaN where either end of the difference is missing. Returned as float32 like the cache.
    """
    length, width = pitch
    x, y = px * length, py * width
    speed = np.full(px.shape, np.nan, dtype=np.float64)
    dx = x[2 * half:] - x[:-2 * half]
    dy = y[2 * half:] - y[:-2 * half]
    speed[half:-half] = np.hypot(dx, dy) / (2 * half / config.FPS)
    return speed.astype(np.float32)


def clean_tracking(game):
    """The clean external game in the cache format of atd.load_cache(), nothing degraded.

    ball gets a third column of NaN (our ATD has a third field, ball height, that the clean
    data does not), ps is computed from positions, every frame is 'seen'.
    """
    n = len(game["ball"])
    ball = np.full((n, 3), np.nan)
    ball[:, :2] = game["ball"]
    return {
        "ball": ball,
        "px": game["px"].copy(),
        "py": game["py"].copy(),
        "ps": speed_from_positions(game["px"], game["py"], game["pitch"]),
        "team": game["team"].copy(),
        "seen": np.ones(n, dtype=bool),
        "home_name": game["name"] + " home",
        "away_name": game["name"] + " away",
        "n_ball_fields": 2,
    }


# --- ball ----------------------------------------------------------------------------------

def fill_short_gaps(ball, max_len):
    """Linearly interpolate NaN runs shorter than max_len that have a visible frame each side."""
    missing = np.isnan(ball[:, 0])
    for start, stop in runs_of(missing):
        if stop - start >= max_len or start == 0 or stop >= len(ball):
            continue
        weights = np.arange(1, stop - start + 1) / (stop - start + 1)
        ball[start:stop] = ball[start - 1] + weights[:, None] * (ball[stop] - ball[start - 1])


def roll_after_stoppage(ball, start, stop, pitch, rng):
    """Step 2: keep following a ball that has left play while it rolls to a stop.

    The clean ball goes NaN a few frames after crossing the line. We continue its last velocity
    with constant deceleration, and keep it visible until it stops plus a short random hold,
    capped at EXIT_MAX_FRAMES, never past the stoppage.
    """
    length, width = pitch
    missing = np.isnan(ball[start:stop, 0])
    if not missing.any():
        return
    first_nan = start + int(np.argmax(missing))
    if first_nan < 3 or np.isnan(ball[first_nan - 3:first_nan, 0]).any():
        return
    velocity = (ball[first_nan - 1] - ball[first_nan - 3]) / 2.0          # per frame, normalised
    vx, vy = velocity[0] * length, velocity[1] * width                     # metres per frame
    speed = np.hypot(vx, vy) * config.FPS                                  # m/s
    stop_time = speed / EXIT_DECEL_MS2                                     # seconds to rest
    n_visible = int(min(stop_time * config.FPS + rng.integers(0, EXIT_HOLD_MAX + 1),
                        EXIT_MAX_FRAMES, stop - first_nan))
    if n_visible <= 0 or speed == 0:
        return
    t = np.minimum(np.arange(1, n_visible + 1) / config.FPS, stop_time)    # seconds
    travelled = speed * t - 0.5 * EXIT_DECEL_MS2 * t * t                   # metres
    ux, uy = vx / np.hypot(vx, vy), vy / np.hypot(vx, vy)
    rows = slice(first_nan, first_nan + n_visible)
    ball[rows, 0] = ball[first_nan - 1, 0] + travelled * ux / length
    ball[rows, 1] = ball[first_nan - 1, 1] + travelled * uy / width


def show_restart_spot(ball, clean_ball, start, stop, pitch, rng):
    """Step 3: in DEAD_SEEN_SHARE of stoppages, the ball is seen lying at the restart spot for
    the final part of the stoppage (a thrower holding it, a free kick being placed)."""
    if rng.random() >= DEAD_SEEN_SHARE or stop >= len(ball):
        return
    share = rng.uniform(0.0, DEAD_SEEN_MAX_FRAC)
    if np.isnan(clean_ball[stop, 0]):
        return                                   # no known restart position
    first = stop - int(round(share * (stop - start)))
    rows = np.arange(first, stop)
    rows = rows[np.isnan(ball[rows, 0])]
    jitter = rng.normal(0.0, 0.05, size=(len(rows), 2)) / np.array(pitch)  # 5 cm
    ball[rows] = clean_ball[stop] + jitter


def degrade_ball(game, labels, rng):
    """Steps 1-4. Returns the degraded (N, 2) ball, normalised, NaN where lost."""
    pitch = game["pitch"]
    length, width = pitch
    clean = game["ball"]
    ball = clean.copy()
    live = (labels == H) | (labels == A)
    dead = labels == D

    fill_short_gaps(ball, FILL_SHORT_GAP)                                      # 1
    for start, stop in runs_of(dead):
        roll_after_stoppage(ball, start, stop, pitch, rng)                     # 2
        show_restart_spot(ball, clean, start, stop, pitch, rng)                # 3

    # 4. Live-play losses. Context comes from the CLEAN ball and players.
    step = np.hypot(np.diff(clean[:, 0]) * length, np.diff(clean[:, 1]) * width) * config.FPS
    ball_speed = np.concatenate(([np.nan], step))
    dist = np.hypot((game["px"] - clean[:, :1]) * length, (game["py"] - clean[:, 1:]) * width)
    with np.errstate(invalid="ignore"):
        crowd = np.sum(dist < CROWD_RADIUS_M, axis=1) >= CROWD_N
        slow = ball_speed < SLOW_BALL_MS
    hazard = per_frame(LOSS_PER_LIVE_MIN) * live
    hazard = hazard * np.where(slow, LOSS_SLOW_MULT, 1.0) * np.where(crowd, LOSS_CROWD_MULT, 1.0)
    ball[bursts(rng, hazard, LOSS_MEDIAN, LOSS_SIGMA)] = np.nan
    return ball


# --- players -------------------------------------------------------------------------------

def nearest_other_distance(px, py, pitch):
    """(N, P) distance in metres from each player to the closest other player; inf if alone."""
    length, width = pitch
    out = np.full(px.shape, np.inf)
    for p in range(px.shape[1]):
        d = np.hypot((px - px[:, p:p + 1]) * length, (py - py[:, p:p + 1]) * width)
        d[:, p] = np.inf
        out[:, p] = np.min(np.where(np.isnan(d), np.inf, d), axis=1)
    return out


def visible_players(game, labels, rng):
    """Steps 5-6: (N, P) bool, True where the tracker sees each real player."""
    on_pitch = ~np.isnan(game["px"])
    n, n_players = on_pitch.shape
    close = nearest_other_distance(game["px"], game["py"], game["pitch"]) < CLOSE_M

    # 6. View-loss episodes hide a random group of players together.
    hazard = per_frame(GROUP_PER_MIN) * np.where(labels == D, GROUP_DEAD_MULT, 1.0)
    starts = np.flatnonzero(rng.random(n) < hazard)
    lengths = lognormal_frames(rng, GROUP_MEDIAN, GROUP_SIGMA, len(starts))
    shares = rng.uniform(*GROUP_FRACTION, size=len(starts))
    members = rng.random((len(starts), n_players)) < shares[:, None]
    hidden = np.zeros_like(on_pitch)
    end = -1
    for start, length, who in zip(starts.tolist(), lengths.tolist(), members):
        if start < end:
            continue
        end = start + length
        hidden[start:end, who] = True

    # 5. Individual losses, more likely when crowded.
    visible = on_pitch & ~hidden
    for p in range(n_players):
        hazard = per_frame(PLAYER_LOSS_PER_MIN) * np.where(close[:, p], CLOSE_MULT, 1.0)
        visible[:, p] &= ~bursts(rng, hazard, PLAYER_LOSS_MEDIAN, PLAYER_LOSS_SIGMA)
    return visible, close


def tracklets_of(visible, close, rng):
    """Step 7: cut each visible run into tracklets at random identity breaks.

    Returns [(start, stop, player, dx_m, dy_m)] with zero offset (duplicates add offsets).
    """
    out = []
    for p in range(visible.shape[1]):
        for start, stop in runs_of(visible[:, p]):
            hazard = per_frame(SWITCH_PER_MIN) * np.where(close[start:stop, p],
                                                          SWITCH_CLOSE_MULT, 1.0)
            cuts = start + np.flatnonzero(rng.random(stop - start) < hazard)
            bounds = [start] + [c for c in cuts.tolist() if c > start] + [stop]
            out.extend((a, b, p, 0.0, 0.0) for a, b in zip(bounds[:-1], bounds[1:]) if b > a)
    return out


def duplicate_tracklets(visible, rng):
    """Step 8: short extra tracks that shadow a visible player at a small offset."""
    n = visible.shape[0]
    out = []
    starts = np.flatnonzero(rng.random(n) < per_frame(DUP_PER_MIN))
    lengths = lognormal_frames(rng, DUP_MEDIAN, DUP_SIGMA, len(starts))
    picks = rng.random(len(starts))
    radii = rng.uniform(*DUP_OFFSET_M, size=len(starts))
    angles = rng.uniform(0, 2 * np.pi, size=len(starts))
    for start, length, pick, radius, angle in zip(starts.tolist(), lengths.tolist(),
                                                  picks, radii, angles):
        candidates = np.flatnonzero(visible[start])
        if not len(candidates):
            continue
        p = int(candidates[int(pick * len(candidates))])
        stop = start + length
        gone = np.flatnonzero(~visible[start:stop, p])     # the shadow ends with the player
        stop = start + int(gone[0]) if len(gone) else min(stop, n)
        out.append((start, stop, p, radius * np.cos(angle), radius * np.sin(angle)))
    return out


def assign_slots(tracklets, team_of_player, rng):
    """Step 9: give every tracklet a random slot of its team that is free at its start.

    Returns (slot_of_tracklet list, team_of_slot array). A team's pool starts at
    SLOTS_PER_TEAM and grows only if every slot is busy.
    """
    free_at = {team: [0] * count for team, count in enumerate(SLOTS_PER_TEAM)}
    slot_ids = {team: list(range(count)) for team, count in enumerate(SLOTS_PER_TEAM)}
    next_id = sum(SLOTS_PER_TEAM)
    order = sorted(range(len(tracklets)), key=lambda i: (tracklets[i][0], tracklets[i][2], i))
    slot_of = [None] * len(tracklets)
    for i in order:
        start, stop, player = tracklets[i][:3]
        team = int(team_of_player[player])
        free = [k for k, t in enumerate(free_at[team]) if t <= start]
        if free:
            k = free[int(rng.integers(len(free)))]
        else:
            free_at[team].append(0)
            slot_ids[team].append(next_id)
            next_id += 1
            k = len(free_at[team]) - 1
        free_at[team][k] = stop
        slot_of[i] = (team, k)
    # Lay the slots out as home first then away, like the ATD metadata.
    columns, team_of_slot = {}, []
    for team in (0, 1):
        for k in range(len(free_at[team])):
            columns[(team, k)] = len(team_of_slot)
            team_of_slot.append(team)
    return [columns[s] for s in slot_of], np.array(team_of_slot, dtype=np.int8)


def degrade(game, labels, replicate=0):
    """The degraded game in atd.load_cache() format. Deterministic for (game, replicate).

    labels (from event_labels.derive_labels) are used ONLY to know live play from stoppages,
    never passed on: the predictor sees tracking only.
    """
    rng = np.random.default_rng([SEED, zlib.crc32(game["name"].encode()), replicate])
    n = len(game["ball"])
    length, width = game["pitch"]

    ball = np.full((n, 3), np.nan)
    ball[:, :2] = degrade_ball(game, labels, rng)

    visible, close = visible_players(game, labels, rng)
    tracklets = tracklets_of(visible, close, rng) + duplicate_tracklets(visible, rng)
    slot_of, team_of_slot = assign_slots(tracklets, game["team"], rng)

    # Speed of each real player from his clean trajectory, so a slot switch is no speed spike.
    speed = speed_from_positions(game["px"], game["py"], game["pitch"])
    px = np.full((n, len(team_of_slot)), np.nan)
    py = np.full((n, len(team_of_slot)), np.nan)
    ps = np.full((n, len(team_of_slot)), np.nan, dtype=np.float32)
    for (start, stop, player, dx, dy), slot in zip(tracklets, slot_of):
        px[start:stop, slot] = game["px"][start:stop, player] + dx / length
        py[start:stop, slot] = game["py"][start:stop, player] + dy / width
        ps[start:stop, slot] = speed[start:stop, player]

    return {
        "ball": ball, "px": px, "py": py, "ps": ps, "team": team_of_slot,
        "seen": np.ones(n, dtype=bool),
        "home_name": game["name"] + " home", "away_name": game["name"] + " away",
        "n_ball_fields": 2,
    }
