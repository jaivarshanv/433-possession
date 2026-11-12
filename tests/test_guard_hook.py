"""The data-guard hook must block reads of match data and allow normal work."""

import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_data.py"


def run_hook(tool, **tool_input):
    """Feed one fake tool call to the hook, return its exit code (0 allow, 2 block)."""
    payload = json.dumps({"tool_name": tool, "tool_input": tool_input})
    return subprocess.run([sys.executable, str(HOOK)], input=payload, text=True,
                          capture_output=True).returncode


def test_blocks_reading_data():
    assert run_hook("Read", file_path="/proj/data/tactical/BAR-ATH_tactical_atd.txt") == 2
    assert run_hook("Read", file_path="/proj/cache/BAR-ATH_tactical_atd.npz") == 2
    assert run_hook("Read", file_path="/anywhere/BAR-ATH_tactical_homography.csv") == 2
    assert run_hook("Grep", pattern="x", path="data") == 2


def test_blocks_shell_viewers_on_data():
    assert run_hook("Bash", command="head -5 data/tactical/BAR-ATH_tactical_atd.txt") == 2
    assert run_hook("Bash", command="python -c 'print(open(\"cache/a.npz\").read())'") == 2
    assert run_hook("Bash", command="cat submissions/v1.csv") == 2


def test_allows_normal_work():
    assert run_hook("Read", file_path="/proj/src/possession/atd.py") == 0
    assert run_hook("Read", file_path="/proj/official/evaluation/gt/BAR-ATH_possession_gt_public.csv") == 0
    assert run_hook("Bash", command="python scripts/01_build_cache.py") == 0
    assert run_hook("Bash", command="ls -la data/tactical") == 0
    assert run_hook("Bash", command="pytest -q") == 0
