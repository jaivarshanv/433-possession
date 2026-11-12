"""Every predictor the harness can score by name. Add new pipeline stages here.

A predictor follows the contract in harness.py: predict(track, info) -> (N,) label codes.
The names are what scripts/12_evaluate_external.py and docs/EXPERIMENTS.md refer to.
"""

from . import baselines, evidence, restarts, training

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
}


# The predictor that produces submissions/433.csv (run.py). Changed only when a new version
# beats it on the external harness by more than the noise (docs/EXPERIMENTS.md).
CHOSEN = "v2_deadball"


def get(name):
    """The predictor registered under `name`, or a readable error listing what exists."""
    if name not in REGISTRY:
        raise KeyError("unknown predictor {!r}; registered: {}".format(name, sorted(REGISTRY)))
    return REGISTRY[name]
