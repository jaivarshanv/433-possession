"""Every predictor the harness can score by name. Add new pipeline stages here.

A predictor follows the contract in harness.py: predict(track, info) -> (N,) label codes.
The names are what scripts/12_evaluate_external.py and docs/EXPERIMENTS.md refer to.
"""

from . import baselines, classifier, evidence, restarts, touches, training

V6_SETTINGS = {"switch_bonus": 1.0, "w_dur": 1.0}     # best of the 13_tune.py grid (D24)


def set_attrs(model, **values):
    """Return `model` with some decoder attributes changed (used by v6)."""
    for key, value in values.items():
        setattr(model, key, value)
    return model


REGISTRY = {
    # The official reference baseline (official/starter/baseline.py), the four published windows.
    "baseline_raw": baselines.nearest_player_predictor(0),
    "baseline_smooth25": baselines.nearest_player_predictor(25),
    "baseline_smooth50": baselines.nearest_player_predictor(50),    # the published floor
    "baseline_smooth125": baselines.nearest_player_predictor(125),
    # M1: rule cells for dead ball + nearest-player possession, HSMM decoder. Cell
    # probabilities, durations and transitions are counted on external labels, leave-one-game-out.
    "v1_rules": training.LogoPredictor(lambda: evidence.RulesModel()),
    # M2: v1 + restart-like kicks in the cells + Laws-of-the-Game restart team (2 passes)
    # + Smart Tagging soft bonus on our match only (fixed a priori, never tuned).
    "v2_deadball": training.LogoPredictor(lambda: restarts.DeadballModel()),
    # ablation: v2 without the Smart Tagging bonus (identical on the external games)
    "v2_deadball_notag": training.LogoPredictor(lambda: restarts.DeadballModel(use_tagging=False)),
    # M3/M4: v2 + every H<->A hand-over snapped to the first new-team touch after the old
    # team's last touch; and the X-policy ablation on top of it.
    "v3_snap": training.LogoPredictor(lambda: touches.SnapModel()),
    "v3_snap_x": training.LogoPredictor(lambda: touches.SnapModel(x_policy=True)),
    # M3 (optional): v3 with gradient-boosted P(dead) and side-symmetric P(home | alive)
    "v4_gbm": training.LogoPredictor(lambda: classifier.GbmModel()),
    "v4_gbm_nosnap": training.LogoPredictor(lambda: classifier.GbmModel(snap=False)),
    # hybrids (ablations): one of the two probabilities from the GBM, the other from v2 cells
    "v4_gbmdead_rulesteam": training.LogoPredictor(
        lambda: classifier.GbmModel(snap=False, gbm_team=False)),
    "v4_rulesdead_gbmteam": training.LogoPredictor(
        lambda: classifier.GbmModel(snap=False, gbm_dead=False)),
    # v5: v4 (no snap) with team shape centred on each team's own half-median, so the
    # possession model reads "unusually spread for this team", not "a wide team" (style)
    "v5_gbm_centred": training.LogoPredictor(
        lambda: classifier.GbmModel(snap=False, centre=True)),
    # v6 (16 Nov): v5 with a cheaper H<->A hand-over and/or duration weight against the
    # measured under-segmentation; values chosen leave-one-game-out (13_tune.py, D24)
    "v6_gbm_switch": training.LogoPredictor(
        lambda: set_attrs(classifier.GbmModel(snap=False, centre=True), **V6_SETTINGS)),
}


# The predictor that produces submissions/433.csv (run.py). Changed only when a new version
# beats it on the external harness by more than the noise (docs/EXPERIMENTS.md).
CHOSEN = "v5_gbm_centred"


def get(name):
    """The predictor registered under `name`, or a readable error listing what exists."""
    if name not in REGISTRY:
        raise KeyError("unknown predictor {!r}; registered: {}".format(name, sorted(REGISTRY)))
    return REGISTRY[name]
