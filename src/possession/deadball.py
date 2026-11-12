"""Dead-ball evidence: is the ball out of play on this frame?

v1 (M1) is a set of plain football rules, each of which puts a frame into one "cell":

  * before the kick-off of a half                          -> certain (the GT is D there)
  * ball located outside the pitch by more than OUT_M       -> strong
  * ball missing, and where it was last / is next seen is outside the pitch -> strong
  * ball missing for a long gap while the players are slow  -> strong (our tracker loses the
    ball mostly during stoppages: 80% missing when players are nearly still)
  * ball located but lying still for a while, players slow  -> medium (free kick being placed)
  * everything else                                         -> live play is likely

Cells cross these conditions with a few bins (gap length, how long the ball has been still,
mean player speed). What each cell MEANS in probability is not guessed: CellTable counts, on the
external games' labels only, how often a frame of that cell is dead ball. The thresholds that
define the cells are round, physically meaningful numbers, listed below.

segments() turns a per-frame dead-ball probability into stoppage segments with hysteresis
(enter / exit thresholds and a minimum length) -- used for reporting and by the restart logic.
"""

import numpy as np

from . import config

OUT_M = 0.5                         # ball centre this far beyond a line = out of play
SPEED_EDGES = (1.0, 1.5, 2.5)       # m/s, mean player speed over 1 s -> 4 bins
GAP_EDGES_S = (0.5, 2.0, 6.0)       # s, length of the ball gap -> 4 bins
STILL_EDGES_S = (1.0, 3.0)          # s, how long the ball has been lying still -> 3 bins

N_SPEED, N_GAP, N_STILL = len(SPEED_EDGES) + 1, len(GAP_EDGES_S) + 1, len(STILL_EDGES_S) + 1
PRESENT_BASE = 1                                       # cell 0 = before kick-off
MISSING_BASE = PRESENT_BASE + 2 * N_STILL * N_SPEED
N_CELLS = MISSING_BASE + 2 * N_GAP * N_SPEED

ENTER_P, EXIT_P, MIN_FRAMES = 0.6, 0.4, 25    # hysteresis for segments()


def speed_bin(f):
    """Mean-player-speed bin per frame (0 = slowest). No players tracked counts as slow."""
    speed = np.nan_to_num(f["speed_mean_1s"], nan=0.0)
    return np.digitize(speed, SPEED_EDGES)


def dead_cells(f, fps=config.FPS):
    """Integer rule cell per frame, 0..N_CELLS-1 (see module docstring)."""
    speed = speed_bin(f)
    present = f["ball_present"]
    out_now = np.nan_to_num(f["ball_out_m"]) > OUT_M
    out_near = (np.nan_to_num(f["last_out_m"]) > OUT_M) | (np.nan_to_num(f["next_out_m"]) > OUT_M)
    still = np.digitize(f["static_len"] / fps, STILL_EDGES_S)
    gap = np.digitize(f["gap_len"] / fps, GAP_EDGES_S)

    present_cell = PRESENT_BASE + (out_now.astype(int) * N_STILL + still) * N_SPEED + speed
    missing_cell = MISSING_BASE + (out_near.astype(int) * N_GAP + gap) * N_SPEED + speed
    cells = np.where(present, present_cell, missing_cell)
    cells[f["pre_kickoff"]] = 0
    return cells.astype(np.int32)


class CellTable:
    """P(hit | cell), counted on training frames and shrunk towards the overall rate.

    With n frames of a cell, h of them hits, and overall rate r:
        p = (h + k * r) / (n + k)          k = PRIOR_FRAMES
    so a cell seen on few frames falls back to the overall rate instead of to 0 or 1.
    Probabilities are clipped to [EPS, 1 - EPS] so their logarithm stays finite.
    """

    PRIOR_FRAMES = 50
    EPS = 1e-3

    def __init__(self, n_cells):
        self.n_cells = n_cells
        self.p = None
        self.rate = None

    def fit(self, cells, hits):
        n = np.bincount(cells, minlength=self.n_cells).astype(np.float64)
        h = np.bincount(cells, weights=hits.astype(np.float64), minlength=self.n_cells)
        rate = h.sum() / max(n.sum(), 1.0)
        self.p = np.clip((h + self.PRIOR_FRAMES * rate) / (n + self.PRIOR_FRAMES),
                         self.EPS, 1 - self.EPS)
        self.rate = float(np.clip(rate, self.EPS, 1 - self.EPS))
        return self

    def predict(self, cells):
        return self.p[cells]


def segments(p_dead, half, enter=ENTER_P, exit_=EXIT_P, min_frames=MIN_FRAMES):
    """Stoppage segments [(start, stop)] from a per-frame P(dead), with hysteresis.

    A stoppage opens when p >= enter and closes when p < exit_; segments shorter than
    min_frames are dropped. Never crosses a half boundary.
    """
    out = []
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        inside, start = False, 0
        for t in rows:
            if not inside and p_dead[t] >= enter:
                inside, start = True, t
            elif inside and p_dead[t] < exit_:
                inside = False
                if t - start >= min_frames:
                    out.append((int(start), int(t)))
        if inside and rows[-1] + 1 - start >= min_frames:
            out.append((int(start), int(rows[-1]) + 1))
    return out
