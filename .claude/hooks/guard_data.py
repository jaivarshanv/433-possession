"""Claude Code PreToolUse hook: keep the match data away from the model.

The competition terms forbid giving the match data (or any extract of it) to a cloud-hosted LLM.
Claude Code is one, so this hook blocks tool calls that would show it the data:

  * Read / Edit / Write / Grep / Glob on a protected path
  * Bash commands that mention a protected path AND use a command that prints file contents

Running our own scripts is allowed (they are written to print aggregates only), and so are
`ls`, `wc`, `du` and `ffprobe` (names, sizes, counts and stream metadata, not content).

How a hook talks to Claude Code: it receives the tool call as JSON on stdin. Exit code 0 lets the
call through; exit code 2 blocks it, and whatever we print to stderr is shown to the model as the
reason.
"""

import json
import re
import sys

# Directories whose contents are match data or frame-level data derived from it.
PROTECTED_DIRS = ("data", "cache", "submissions")

# File name endings that are match assets wherever they live.
PROTECTED_SUFFIXES = ("_atd.txt", "_atd.xml", "_homography.csv", "_smart_tagging.xml", ".mp4")

# A path segment like "data/..." at the start of the string or after a space, quote, slash or =.
DIR_PATTERN = re.compile(r'(^|[\s"\'=/(])(' + "|".join(PROTECTED_DIRS) + r')(/|$|\s)')

# Shell commands that print or copy file contents. If one of these appears in a command together
# with a protected path, the command is blocked.
VIEWERS = (
    "cat", "head", "tail", "less", "more", "bat", "nl", "xxd", "od", "hexdump", "strings",
    "grep", "rg", "ag", "awk", "sed", "cut", "sort", "uniq", "column", "jq", "csvlook",
    "python -c", "python3 -c", "ipython", "node -e", "perl", "ruby",
    "ffplay", "ffmpeg", "open", "qlmanage", "cp", "mv", "scp", "rsync", "curl", "nc", "base64",
    "zip", "tar", "git add",
)


def mentions_protected(text):
    """True if `text` (a path or a whole command) refers to match data."""
    if not text:
        return False
    if DIR_PATTERN.search(text):
        return True
    lowered = text.lower()
    return any(suffix in lowered for suffix in PROTECTED_SUFFIXES)


def uses_viewer(command):
    """True if the shell command uses something that would print or move file contents."""
    for viewer in VIEWERS:
        # Match the viewer as a whole word, e.g. "cat" but not "concatenate".
        if re.search(r'(^|[\s;|&(`])' + re.escape(viewer) + r'(\s|$)', command):
            return True
    return False


def block(reason):
    """Refuse the tool call and tell the model why."""
    sys.stderr.write(
        "BLOCKED by .claude/hooks/guard_data.py: " + reason + "\n"
        "Match data must never be shown to the model (competition terms). "
        "Write a script that prints aggregates only, and let the human run/inspect it. "
        "See CLAUDE.md section 0.\n")
    sys.exit(2)


def main():
    call = json.load(sys.stdin)
    tool = call.get("tool_name", "")
    args = call.get("tool_input", {}) or {}

    if tool in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = args.get("file_path") or args.get("notebook_path") or ""
        if mentions_protected(path):
            block("{} on protected path {}".format(tool, path))

    elif tool in ("Grep", "Glob"):
        target = " ".join(str(args.get(key, "")) for key in ("path", "pattern", "glob"))
        if tool == "Grep":
            target = " ".join(str(args.get(key, "")) for key in ("path", "glob"))
        if mentions_protected(target):
            block("{} over protected location".format(tool))

    elif tool == "Bash":
        command = args.get("command", "")
        if mentions_protected(command) and uses_viewer(command):
            block("shell command would read or copy match data")

    sys.exit(0)  # everything else is allowed


if __name__ == "__main__":
    main()
