"""Thin wrapper around the official metric library (official/evaluation/possession_metrics.py).

We never re-implement the metric: the official file is the specification, and using it directly
means our numbers are exactly the numbers the organisers will compute.

Two uses:
  1. Check a submission against the PUBLIC 5 minutes -- pipeline correctness only.
  2. Score predictions on EXTERNAL games against labels we derived from their event data.
     This is where all model selection happens. score_codes() takes any GT array.
"""

import numpy as np

from . import config


def _pm():
    """Import the official metric module lazily (needs official/ on sys.path)."""
    config.add_official_to_path()
    import possession_metrics
    return possession_metrics


def labels_to_codes(labels):
    """Array of 'H'/'A'/'D'/'X' strings -> int8 codes 0..3 (official mapping)."""
    return _pm().label_codes(np.asarray(labels, dtype=object))


def load_public_gt_codes():
    """The public ground truth as codes. Everything outside the 5-minute window is X."""
    pm = _pm()
    return pm.label_codes(pm.read_label_csv(str(config.PUBLIC_GT)))


def score_codes(gt_codes, pred_codes, bootstrap_n=0):
    """Full official evaluation of one prediction against one ground truth (same length).

    bootstrap_n > 0 adds the official block-bootstrap 5-95% intervals (seeded, reproducible).
    """
    return _pm().evaluate(np.asarray(gt_codes), np.asarray(pred_codes), bootstrap_n=bootstrap_n)


def summarize(result):
    """The handful of numbers worth reading, from an evaluate() result."""
    return {
        "spellF1": round(result["spell_f1_mean"], 4),
        "macroF1": round(result["macro_f1"], 4),
        "alive_dead_F1": round(result["alive_dead_f1"], 4),
        "F1_H": round(result["per_class"]["H"]["f1"], 4),
        "F1_A": round(result["per_class"]["A"]["f1"], 4),
        "F1_D": round(result["per_class"]["D"]["f1"], 4),
        "pred_possessions": result["n_pred_spells"],
        "gt_possessions": result["n_gt_spells"],
        "pred_mean_s": round(result["mean_spell_duration_s"], 2),
        "gt_mean_s": round(result["gt_mean_spell_duration_s"], 2),
    }


def print_summary(name, result, public=False):
    """Print a one-block summary. Aggregates only, safe to show anyone."""
    s = summarize(result)
    tag = "  [PUBLIC 5-MIN WINDOW: pipeline check only, do not tune on this]" if public else ""
    print("{}{}".format(name, tag))
    print("  spellF1 {spellF1:.4f}   macroF1 {macroF1:.4f}   alive/dead F1 {alive_dead_F1:.4f}"
          .format(**s))
    print("  per-class F1   H {F1_H:.4f}   A {F1_A:.4f}   D {F1_D:.4f}".format(**s))
    print("  possessions    pred {pred_possessions} (mean {pred_mean_s} s)   "
          "gt {gt_possessions} (mean {gt_mean_s} s)".format(**s))
