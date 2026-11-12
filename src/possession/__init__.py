"""Ball-possession inference for MTAH 2027 Phase 1.

Modules:
    config      paths and match constants (the only place they live)
    atd         parse the automatic tracking data into numpy arrays, cached on disk
    frames      graded-frame masks, half ids, run-length helpers
    tagging     Smart Tagging intervals as a DataFrame
    scoring     wrapper around the official metric library
    submission  write and validate submission files
    baselines   the official nearest-player baseline on any tracking dict

External validation (stage 2, docs/EXTERNAL_VALIDATION.md):
    metrica       download + load Metrica's open sample games
    event_labels  frame-level H/A/D labels from an external game's events
    degrade       make clean external tracking look like our ATD
    profile       aggregate tracking statistics, to calibrate degrade
    harness       predictor contract, scoring on degraded external games
    predictors    registry of named predictors
"""
