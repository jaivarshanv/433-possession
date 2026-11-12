"""Harness diagnostics beyond the official metric: stoppages, restarts, boundaries.

Run on EXTERNAL games only (they have labels); every number is an aggregate. They explain a
spellF1 change instead of just measuring it:

  stoppages   how many D stretches were predicted vs true, how many true ones were found, and
              how many frames early/late their starts and ends are (median, and median |error|)
  restarts    after each true stoppage: did we give the ball to the right team?
"""

import numpy as np

from . import config, decoder

H, A, D, X = config.H, config.A, config.D, config.X
RESTART_CHECK = 25      # judge the restart team one second after the true restart


def dead_segments(codes):
    """[(start, stop)] of the D runs inside graded frames."""
    return [(a, b) for state, a, b in decoder.segments_of(codes) if state == D]


def stoppage_errors(gt, pred):
    """(n_true, n_pred, n_found, start errors, end errors) in frames, + = predicted late.

    Each true stoppage is matched to the not-yet-used predicted stoppage it overlaps most.
    """
    true_d, pred_d = dead_segments(gt), dead_segments(pred)
    starts, ends, used = [], [], set()
    for a, b in true_d:
        best, best_overlap = None, 0
        for j, (c, d) in enumerate(pred_d):
            overlap = min(b, d) - max(a, c)
            if overlap > best_overlap and j not in used:
                best, best_overlap = j, overlap
        if best is not None:
            used.add(best)
            starts.append(pred_d[best][0] - a)
            ends.append(pred_d[best][1] - b)
    return len(true_d), len(pred_d), len(used), starts, ends


def restart_counts(gt, pred, check=RESTART_CHECK):
    """(checked, right): true restarts, and those after which the predicted team is right
    `check` frames later. Restarts followed by a possession shorter than that are skipped."""
    right = total = 0
    for _, b in dead_segments(gt):
        frame = b + check
        if frame >= len(gt) or gt[b] not in (H, A) or np.any(gt[b:frame + 1] != gt[b]):
            continue
        total += 1
        right += int(pred[frame] == gt[b])
    return total, right


def report(pairs):
    """Pooled diagnostics over [(gt, pred)] pairs of external games."""
    n_true = n_pred = n_found = checked = right = 0
    starts, ends = [], []
    for gt, pred in pairs:
        t, p, f, s, e = stoppage_errors(gt, pred)
        n_true, n_pred, n_found = n_true + t, n_pred + p, n_found + f
        starts += s
        ends += e
        c, r = restart_counts(gt, pred)
        checked, right = checked + c, right + r
    starts, ends = np.array(starts), np.array(ends)

    def med(x):
        return round(float(np.median(x)), 1) if len(x) else float("nan")

    return {
        "gt_stoppages": n_true, "pred_stoppages": n_pred, "found": n_found,
        "start_err_median": med(starts), "start_abs_median": med(np.abs(starts)),
        "end_err_median": med(ends), "end_abs_median": med(np.abs(ends)),
        "restart_team_right": round(right / checked, 3) if checked else float("nan"),
    }
