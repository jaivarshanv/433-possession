"""Frame-level H / A / D ground truth for an external game, derived from its event data.

The official ground truth is "mechanical from corrected tracking": D where the corrected tracking
marks the ball dead, otherwise the team that owns the ball, and the owner changes only when the
other team gains CONTROLLED possession (CHALLENGE.md). Metrica's sample games give us exactly
the two ingredients to imitate that on other matches:

  * corrected tracking, whose ball is NaN almost exactly while the ball is out of play
    (it reappears on the very frame of the restart -- measured, median offset 0 frames), and
  * hand-made event data: passes, recoveries, duels, set pieces, balls out, fouls.

The rules, in order (every constant is named below, none refers to a specific frame):

DEAD BALL (label D) = from the stoppage to the restart.
  1. Before the kick-off of each period: D.
  2. A stretch where the clean ball is NaN is a stoppage if ANY of
       - a restart event (any SET PIECE, or a goal kick) starts inside it or within
         RESTART_TOL frames of its edges,
       - a stoppage event (ball out, foul, offside, hand ball, end of half, a shot that went
         out or in) happens within STOP_TOL frames before it starts,
       - the ball was outside the pitch just before the NaN began,
       - it is at least LONG_DEAD frames long (corrected tracking does not lose a live ball
         for that long), or it runs to the end of the period.
     Otherwise it is a tracking hole during live play (2-4 s scrambles exist) and stays alive.
  3. The stoppage starts where the ball crossed the line, if it was seen outside the pitch
     just before the NaN (walk back over the out-of-bounds frames), else where the NaN starts.
     It ends at the restart event, or where the ball reappears if no restart was tagged.
  4. A restart with no NaN stretch before it (the corrected tracking kept the ball during a
     quick free kick) is D from the ball crossing the line, or from the end of the last event
     before it, up to the restart.

POSSESSION (labels H / A) inside each live stretch.
  5. The live stretch starts with the restarting team.
  6. Possession passes to the other team only at a CONTROLLED touch of that team:
       - a restart it takes,
       - the reception of a completed pass (Metrica's PASS is always completed: To = receiver),
       - a recovery / won duel / carry by a player who then KEEPS the ball for at least
         MIN_HOLD frames before his next action (or before the opponent touches it).
     A one-touch recovery (Metrica writes RECOVERY and BALL LOST on the same frame), a header
     won in the air, a deflection: none of these change possession. A loose ball stays with
     the last team that controlled it.
"""

from collections import Counter

import numpy as np

from . import config

# --- named constants (frames at 25 fps) ------------------------------------------------------

RESTART_TOL = 25        # a restart within 1 s of a NaN stretch's edges belongs to it
STOP_TOL = 50           # a stoppage event up to 2 s before the NaN starts explains it
LONG_DEAD = 250         # a NaN stretch of 10 s or more is a stoppage whatever the events say
MIN_HOLD = 3            # a touch kept >= 3 frames (0.12 s) is control; same-frame is one-touch

# Event types that end a player's time on the ball.
ACTIONS = ("PASS", "BALL LOST", "SHOT", "BALL OUT")
# Event types / subtype fragments that stop play.
STOP_TYPES = ("BALL OUT", "FAULT RECEIVED")
STOP_SUBTYPES = ("OUT", "GOAL", "FAULT", "OFFSIDE", "HAND BALL", "END HALF")

H, A, D, X = config.H, config.A, config.D, config.X
TEAM_CODE = {"H": H, "A": A}


def runs_of(mask):
    """(start, stop) pairs of the True runs of a boolean array, stop exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def is_restart(events):
    """Boolean Series: the event restarts play (set piece, goal kick, kick-off)."""
    return (events["type"] == "SET PIECE") | events["subtype"].str.contains("GOAL KICK")


def is_stoppage(events):
    """Boolean Series: the event stops play (ball out, foul, offside, goal, end of half).

    Duel subtypes such as "TACKLE-ADVANTAGE-WON" are NOT stoppages: the referee let play go on.
    A goal kick ("GOAL KICK" contains "GOAL") is a restart, not a stoppage.
    """
    sub = events["subtype"].str.replace("GOAL KICK", "", regex=False)
    by_subtype = sub.apply(lambda text: any(part in text for part in STOP_SUBTYPES))
    return (events["type"].isin(STOP_TYPES) | by_subtype) & ~sub.str.contains("ADVANTAGE")


def ball_outside(ball):
    """Boolean per frame: the (clean) ball is located outside the 0..1 pitch box."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < 0) | (x > 1) | (y < 0) | (y > 1)


def stop_moment(event):
    """When a stoppage event stopped play: a foul or hand ball when it happens (event start),
    anything else when the action ends (a ball out or a shot reaching the line or the net)."""
    text = event["type"] + " " + event["subtype"]
    return int(event["start"]) if ("FAULT" in text or "HAND BALL" in text) else int(event["end"])


def stoppage_frame(event, outside, limit):
    """The frame play stopped because of one stoppage event (used by rule 4).

    The first frame the ball is seen outside the pitch between the event and `limit` (it
    crossed the line), else stop_moment().
    """
    start = int(event["start"])
    window = outside[start:limit]
    if window.any():
        return start + int(np.argmax(window))
    return stop_moment(event)


def crossing_frame(outside, first_nan, earliest):
    """Walk back from a NaN stretch over the frames where the ball was outside the pitch.

    Returns the first frame of that out-of-bounds run (the moment the ball crossed the line), or
    first_nan if the ball was not seen outside just before it. Never goes before `earliest`.
    """
    frame = first_nan
    while frame > earliest and outside[frame - 1]:
        frame -= 1
    return frame


def dead_stretches(game, first, last, counts):
    """List of (start, stop, restart_team) dead-ball stretches of one period, stop exclusive.

    first, last = array rows of the period, inclusive. restart_team is "H"/"A", or None when no
    restart event was tagged. counts (a Counter) records which rule fired, for reporting.
    """
    events = game["events"]
    ev = events[(events["start"] >= first) & (events["start"] <= last)]
    restarts = ev[is_restart(ev)]
    stops = ev[is_stoppage(ev)]
    stop_frames = np.array([stop_moment(event) for _, event in stops.iterrows()], dtype=np.int64)
    outside = ball_outside(game["ball"])
    missing = np.isnan(game["ball"][:, 0])

    kickoffs = restarts[restarts["subtype"].str.startswith("KICK OFF")]
    kickoff = int(kickoffs["start"].iloc[0]) if len(kickoffs) else first
    out = []
    if kickoff > first:
        out.append((first, kickoff, None))                                    # rule 1
        counts["pre_kickoff"] += 1

    covered = set()   # restart events already used as the end of a NaN stretch
    for a, b in runs_of(missing[kickoff:last + 1]):
        a, b = a + kickoff, b + kickoff
        near = restarts[(restarts["start"] >= a - RESTART_TOL)
                        & (restarts["start"] <= b + RESTART_TOL)]
        if len(near):
            reason = "restart_event"
        elif np.any((stop_frames >= a - STOP_TOL) & (stop_frames < b)):
            # play stopped shortly before the ball vanished, or while it was gone
            reason = "stoppage_event"
        elif a > first and outside[a - 1]:
            reason = "ball_left_pitch"                     # last seen outside, then gone
        elif b - a >= LONG_DEAD:
            reason = "long_gap"
        elif b > last:
            reason = "end_of_period"
        else:
            counts["live_tracking_hole"] += 1                                  # rule 2, alive
            continue
        counts[reason] += 1
        start = crossing_frame(outside, a, kickoff)                            # rule 3
        if len(near):
            covered.update(near.index.tolist())
            # The restart can come a few frames before the ball reappears; it ends the stoppage.
            stop, team = max(int(near["start"].iloc[-1]), start), near["team"].iloc[-1]
        else:
            stop, team = b, None
        out.append((start, min(stop, last + 1), team))

    # Rule 4: restarts the corrected tracking never lost the ball for. Look only at what
    # happened since the previous restart: the LAST stoppage event there stopped play (earlier
    # ones were followed by play, e.g. advantage). With no stoppage event, the end of the last
    # action. With no event at all since the previous restart, it is a duplicate tag: skip.
    play = ev[(ev["type"] != "CARD") & ~is_restart(ev)]
    restart_frames = np.sort(restarts["start"].values)
    for index, restart in restarts.iterrows():
        r = int(restart["start"])
        if index in covered or r <= kickoff:
            continue
        previous = restart_frames[restart_frames < r]
        since = play[(play["start"] < r) & (play["start"] > (previous[-1] if len(previous) else kickoff))]
        if not len(since):
            counts["duplicate_restart_tag"] += 1
            continue
        stopping = since[is_stoppage(since)]
        if len(stopping):
            start = stoppage_frame(stopping.iloc[-1], outside, r)
        else:
            start = stoppage_frame(since.iloc[-1], outside, r) + 1
        if start < r:
            out.append((start, r, restart["team"]))
            counts["restart_without_gap"] += 1
    return sorted(out)


def control_moments(events, stretch_start, stretch_stop):
    """[(frame, team_code)] of controlled touches inside one live stretch, sorted by frame.

    Implements rule 6. Only events starting inside [stretch_start, stretch_stop) are used, so
    nothing that happens during dead time (cards, the keeper fetching the ball) counts.
    """
    ev = events[(events["start"] >= stretch_start) & (events["start"] < stretch_stop)]
    moments = []

    for row in ev[is_restart(ev)].itertuples():
        moments.append((row.start, TEAM_CODE[row.team]))

    for row in ev[ev["type"] == "PASS"].itertuples():
        if row.end < stretch_stop:
            moments.append((row.end, TEAM_CODE[row.team]))                    # reception

    won_duel = (ev["type"] == "CHALLENGE") & ev["subtype"].str.contains("WON")
    pickups = ev[ev["type"].isin(("RECOVERY", "CARRY")) | (won_duel & ~ev["subtype"].str.contains("FAULT"))]
    actions = ev[ev["type"].isin(ACTIONS)]
    touches = ev[ev["type"].isin(ACTIONS + ("RECOVERY", "CARRY", "SET PIECE")) | won_duel]
    for row in pickups.itertuples():
        # How long did this player keep it? Until his own next action, or until the other team
        # touched the ball, or until play stopped -- whichever came first.
        own = actions[(actions["player"] == row.player) & (actions["start"] >= row.start)]
        rival = touches[(touches["team"] != row.team) & (touches["start"] >= row.start)]
        ends = [stretch_stop]
        if len(own):
            ends.append(int(own["start"].iloc[0]))
        if len(rival):
            ends.append(int(rival["start"].iloc[0]))
        if min(ends) - row.start >= MIN_HOLD:
            moments.append((row.start, TEAM_CODE[row.team]))
    return sorted(moments)


def derive_labels(game):
    """H/A/D/X codes for every array row of an external game, plus a report dict.

    X outside the two periods only. Inside them every frame is H, A or D -- pre-kickoff frames
    are D, like the first 2934 graded frames of our match's second half.
    The report holds counts only: how many stoppages each rule found.
    """
    n = len(game["ball"])
    codes = np.full(n, X, dtype=np.int8)
    counts = Counter()
    events = game["events"]

    for period in (1, 2):
        rows = np.flatnonzero(game["period"] == period)
        first, last = int(rows[0]), int(rows[-1])
        dead = np.zeros(n, dtype=bool)
        restart_team = {}
        for start, stop, team in dead_stretches(game, first, last, counts):
            dead[start:stop] = True
            if team is not None:
                restart_team[stop] = TEAM_CODE[team]

        live = ~dead
        live[:first] = False
        live[last + 1:] = False
        for start, stop in runs_of(live):
            moments = control_moments(events, start, stop)
            # Rule 5: the restarting team, else the first team to control the ball.
            team = restart_team.get(start, moments[0][1] if moments else None)
            if team is None:
                counts["live_stretch_without_events"] += 1
                team = codes[start - 1] if start > 0 and codes[start - 1] in (H, A) else H
            codes[start:stop] = team
            for frame, owner in moments:
                if owner != team:
                    codes[frame:stop] = owner
                    team = owner
        codes[first:last + 1][dead[first:last + 1]] = D
    return codes, dict(counts)
