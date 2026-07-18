# P3 — VLM & Orchestrator (Track A, must ship)

**Owns:** VLM prompt → strict JSON, parsing, item→destination policy, closed-loop state machine (home → photo → plan → pick → drop → repeat), retries/skips, decision logging for the judge screen.

**Can build against phone photos before the arm exists.** Use `shared.fixtures` when P1/P2 are late.

## Contract

- **In from P2:** `pixel_to_arm` + centroid refine  
- **Out to P1:** `(arm_xy_mm, category)` → `pick_and_drop`

## Destinations (constrained)

`trash | pen_cup | tray | keep` — open vocab on labels, closed on destinations. Never touch `keep`.

## Saturday exit

One fully autonomous single-object cycle: photo → destination, no human input. Then multi-object closed loop (6-object clean unattended).

## Demo UI

Every photo + JSON decision logged under `runs/<timestamp>/`. Keep that window visible for judges.
