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
| D9 | Team name + public repo | — | needed for `<team>.csv` / `<team>.pdf` naming | OPEN (by Nov 10) |
