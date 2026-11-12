"""Who restarts play after a stoppage -- Laws of the Game -- and the v2 dead-ball model (M2).

After every predicted stoppage we look at the restart: the first restart-like kick near the
end of the stoppage (features.restart_kicks), and where the ball was when it was kicked.
The position says which restart it is, and the Laws say who takes it:

  corner kick   ball within CORNER_M of a corner flag  -> the team ATTACKING that goal line
  goal kick     ball in (or just around) a goal area   -> the team DEFENDING that goal
  kick-off      ball near the centre spot, and the stoppage began with the ball vanishing
                into a goal mouth (a goal)              -> the team that CONCEDED
  throw-in      ball on a touchline                     -> the team of the player at the ball
  other         free kick, drop ball ...                -> the team of the player at the ball
                                                           (opponents must be 9.15 m away)

("Throw-in to the team that did not touch last" is the Law, but who touched last is measured
badly through our degraded tracking -- 45% agreement on the external games, i.e. a coin
flip -- whereas the thrower standing at the ball is seen directly.)

How much to trust each rule is not assumed: on the external training games, the model counts
how often each rule names the team that really restarts (CellTable). The decoder then gets a
bonus for that team during the RESTART_BONUS_S after the kick: the log-odds of the rule,
times a weight chosen on the harness.

v2 decodes twice: once to find the stoppages, then again with the restart evidence added.
Smart Tagging (our match only -- the external games have none) adds a soft, fixed bonus for
the team a SET PIECES instance names; its size TAG_TOTAL_LOG is set a priori and never tuned.
"""

import numpy as np

from . import config, deadball, decoder, evidence

H, A, D, X = config.H, config.A, config.D, config.X

CORNER_M = 3.0                  # ball within 3 m of a corner flag
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_M = 5.5, 9.16   # Laws: goal area 5.5 m x 18.32 m
GOAL_AREA_TOL_M = 2.0           # tracking / placement tolerance around the goal area
CENTRE_M = 3.0                  # ball within 3 m of the centre spot
TOUCHLINE_M = 1.5               # ball within 1.5 m of a touchline
GOAL_MOUTH_HALF_M = 3.66 + 1.5  # half the goal width (3.66 m) + tolerance
GOAL_LINE_M = 3.0               # "vanished at the goal line": within 3 m of it
SEARCH_BEFORE_S, SEARCH_AFTER_S = 1.0, 2.0     # look for the restart kick around the D end
RESTART_BONUS_S = 1.0           # how long after the kick the restart evidence counts
CHECK_FRAMES = 12               # rule judged right if the true label 0.5 s after is its team
TAG_TOTAL_LOG = 1.0             # Smart Tagging: at most 1 nat for a tagged set piece (a priori)

KINDS = ("corner", "goal_kick", "kick_off", "throw_in", "other")
W_RESTART = 1.0                 # default weight of the restart evidence (tuned on the harness)


def attacker_of_end(home_dir_value, end_is_far):
    """Team (0 home / 1 away) attacking the goal at x = length (end_is_far) or at x = 0."""
    home_attacks_far = home_dir_value > 0
    return 0 if home_attacks_far == end_is_far else 1


def restart_kind(x, y, length, width, vanished_in_goal_mouth):
    """Which restart a ball position looks like (one of KINDS)."""
    to_goal_line = min(abs(x), abs(x - length))
    to_touchline = min(abs(y), abs(y - width))
    if to_goal_line < CORNER_M and to_touchline < CORNER_M:
        return "corner"
    if (to_goal_line < GOAL_AREA_DEPTH_M + GOAL_AREA_TOL_M
            and abs(y - width / 2) < GOAL_AREA_HALF_M + GOAL_AREA_TOL_M):
        return "goal_kick"
    if np.hypot(x - length / 2, y - width / 2) < CENTRE_M and vanished_in_goal_mouth:
        return "kick_off"
    if to_touchline < TOUCHLINE_M:
        return "throw_in"
    return "other"


def find_restarts(f, codes, info):
    """[(kind, team, frame)] for every predicted stoppage that ends inside a half.

    frame = the restart kick (or the end of the predicted stoppage if no kick is found);
    team = the rule's team (0 home / 1 away), or -1 when the rule cannot tell.
    """
    length, width = info["pitch"]
    fps = info["fps"]
    kicks = np.flatnonzero(f["kick_since"] == 0)
    out = []
    for state, start, stop in decoder.segments_of(codes):
        if state != D or stop >= len(codes) or codes[stop] == X:
            continue
        lo, hi = stop - SEARCH_BEFORE_S * fps, stop + SEARCH_AFTER_S * fps
        near = kicks[(kicks >= lo) & (kicks <= hi)]
        frame = int(near[np.argmin(np.abs(near - stop))]) if len(near) else int(stop)
        x, y = f["kick_x"][frame], f["kick_y"][frame]
        if not len(near):                               # no kick: first sight of the ball
            seen = np.flatnonzero(f["ball_present"][stop:int(hi)])
            if not len(seen):
                out.append(("other", -1, frame))
                continue
            x, y = f["bx"][stop + seen[0]], f["by"][stop + seen[0]]
        # did the stoppage start with the ball vanishing into a goal mouth? (a goal)
        lx, ly = f["last_bx"][start], f["last_by"][start]
        in_mouth = bool(np.isfinite(lx) and min(abs(lx), abs(lx - length)) < GOAL_LINE_M
                        and abs(ly - width / 2) < GOAL_MOUTH_HALF_M)
        kind = restart_kind(x, y, length, width, in_mouth)
        home_dir = f["home_dir"][frame]
        if kind == "corner":
            team = attacker_of_end(home_dir, abs(x - length) < abs(x))
        elif kind == "goal_kick":
            team = 1 - attacker_of_end(home_dir, abs(x - length) < abs(x))
        elif kind == "kick_off":
            team = 1 - attacker_of_end(home_dir, abs(lx - length) < abs(lx))   # conceded
        else:                                            # the team of the player at the ball
            d_home = np.nan_to_num(f["d_home"][frame], nan=np.inf)
            d_away = np.nan_to_num(f["d_away"][frame], nan=np.inf)
            team = -1 if min(d_home, d_away) > evidence.GATE_M else int(d_away < d_home)
        out.append((kind, int(team), frame))
    return out


def restart_bonus(found, table, n, w_restart, fps=config.FPS):
    """(n, 3) score bonus: log-odds of each restart rule for its team, spread over the bonus span."""
    bonus = np.zeros((n, 3))
    span = int(RESTART_BONUS_S * fps)
    for kind, team, frame in found:
        if team < 0:
            continue
        p = table.p[KINDS.index(kind)]
        bonus[frame:frame + span, team] += w_restart * np.log(p / (1 - p)) / span
    return bonus


def tagging_bonus(tags, n):
    """(n, 3) Smart Tagging bonus: TAG_TOTAL_LOG spread over each team-tagged SET PIECES phase."""
    bonus = np.zeros((n, 3))
    for start, stop, team in tags:
        if stop > start and 0 <= start < n:
            bonus[start:min(stop, n), team] += TAG_TOTAL_LOG / (stop - start)
    return bonus


class DeadballModel(evidence.RulesModel):
    """v2: v1 + restart-like kicks in the cells + Laws-of-the-Game restart evidence (2 passes)."""

    n_dead_cells = deadball.N_CELLS_V2
    n_team_cells = evidence.N_TEAM_CELLS_V2

    def __init__(self, w_dead=evidence.W_DEAD, w_team=evidence.W_TEAM, w_restart=W_RESTART,
                 use_tagging=True):
        super().__init__(w_dead, w_team)
        self.w_restart = w_restart
        self.use_tagging = use_tagging

    def dead_cells(self, feats):
        return deadball.dead_cells_v2(feats)

    def team_cells(self, feats):
        return evidence.team_cells_v2(feats)

    def fit(self, train):
        super().fit(train)
        # How often does each restart rule name the right team? Judged on the training games'
        # own first-pass stoppages, exactly as the rules will be used on a new game.
        kinds, hits = [], []
        for feats, info, gt in train:
            first_pass = self.decode(self.scores(feats), feats)
            for kind, team, frame in find_restarts(feats, first_pass, info):
                check = frame + CHECK_FRAMES
                if team >= 0 and check < len(gt) and gt[check] in (H, A):
                    kinds.append(KINDS.index(kind))
                    hits.append(gt[check] == team)
        self.restart_table = deadball.CellTable(len(KINDS)).fit(
            np.array(kinds, dtype=np.int64), np.array(hits, dtype=bool))
        return self

    def predict(self, feats, info):
        scores = self.scores(feats)
        if self.use_tagging and info.get("tagging"):
            scores = scores + tagging_bonus(info["tagging"], len(scores))
        first_pass = self.decode(scores, feats)
        found = find_restarts(feats, first_pass, info)
        scores = scores + restart_bonus(found, self.restart_table, len(scores), self.w_restart)
        return self.decode(scores, feats)
