# Decisions

Status: OPEN / DECIDED (date, reason). Update when one is made.

| # | decision | options | recommendation | status |
|---|---|---|---|---|
| D1 | Solo vs team | stay solo; recruit 2-4 | ask organisers whether solo is ranked in Phase 1 (teams <3 excluded from Innovation Challenge / Final Pitch) | OPEN — currently solo |
| D2 | Data terms vs AI assistants | aggregates-only workflow; written permission | aggregates-only until written answer; enforced by `.claude/hooks/guard_data.py` | OPEN — aggregates-only in force |
| D3 | External training data | Metrica samples; PFF WC 2022; SkillCorner | start Metrica samples, add PFF for volume | Metrica samples IN USE as the validation set (stage 2, 2026-10-30; 3 games, 759 possessions). PFF for training volume still OPEN |
| D4 | Smart Tagging use | ignore; soft prior; hard anchors | soft prior + restart anchors, never copied as labels | OPEN |
| D5 | Own ball detector | yes / no | decide after v2 on time left + measured inference speed | OPEN (by Nov 5) |
| D6 | Whistle detection | yes / no | audio track exists in both videos, tactical in sync | OPEN — feasible, stretch |
| D7 | Broadcast sync | yes / no | last priority, only if ahead after v2 | OPEN (by Nov 6) |
| D8 | Sequence model | HSMM; learned BiGRU/TCN; both | HSMM first; learned only if better on external val | OPEN |
| D9 | Team name + public repo | — | needed for `<team>.csv` / `<team>.pdf` naming | Team name DECIDED: **433** (files `433.csv`, `433.pdf`). Repo: see D10 |
| D10 | GitHub repo `SkAztecs/433-possession` (private) | create + push now; keep local | create + push before any code (brief STEP 0) | NOT DONE (12 Nov): the `gh repo create ... --push` call was refused by Claude Code's permission classifier, and `gh` is logged in as `jaivarshanv`, a different user from `SkAztecs` (a user account, not an org), so the command could not work as written anyway. Safest reversible option taken: everything committed locally on `main`, nothing pushed. A human must log in as the right account and run the create/push command (final report) |
| D12 | v1 decoder weights | grid 0.03-0.4 for w_dead, w_team | equal weights 0.1 | DECIDED 12 Nov (13_tune.py): (0.1, 0.1) best in-sample 0.636 and picked by 2 of 3 LOGO folds; (0.05, 0.1) within noise. Kept the equal round value: one shared weight is one fewer free parameter |
| D13 | 433.csv after M1 | keep baseline; v1_rules | v1_rules | DECIDED 12 Nov: +0.30 external spellF1 over baseline_smooth50 (far above noise); public check also above baseline on every metric |
| D11 | How learned parts are validated | fit on all 3 external games; leave-one-game-out (LOGO) | LOGO: anything fitted from external labels (duration priors, rule-cell probabilities, classifier) is fitted on the two OTHER games when the harness scores a game; our match uses all three | DECIDED 12 Nov: in-sample scores would flatter learned versions against the rule-based ones |
