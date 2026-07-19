# P3 — VLM & Orchestrator (Track A, must ship)

**Owns (per GROUND_TRUTH):** VLM prompt → strict JSON, parsing, item→destination policy, closed-loop state machine, retry/skip, decision logging for the judge screen.

**Can build against phone photos before the arm exists.** Use `shared.fixtures` when P1/P2 are late.

---

## Contracts (GT §2.6)

| Direction | Contract |
|-----------|----------|
| In from P2 | `pixel → arm_mm` (`CalibrationChain`) + `centroid_refine` inside VLM box |
| Out to P1 | `pick_and_drop(..., target_xy_mm, item_type, destination_name)` |

## Destinations (constrained)

`trash | pen_cup | tray | keep` — open-vocab labels, closed destinations.  
**Never touch `keep`** (judge phone beat).

## Closed loop (GT §2.5)

`home → photo → plan → pick → drop → repeat`  
- Photos **only from home** (arm out of FOV).  
- Max **2 retries** per object, then skip.  
- Failures self-heal on the next photo.

## Run

```bash
# dry / fixtures (no arm, no API)
python -m p3_vlm_orchestrator.run_clean --fixtures

# live VLM (needs config/.env keys + calibrated arm)
DESKPARTNER_DRY_RUN=0 python -m p3_vlm_orchestrator.run_clean --perception vlm

# wifi/API dead
DESKPARTNER_PERCEPTION=cv DESKPARTNER_DRY_RUN=0 python -m p3_vlm_orchestrator.run_clean
```

Fallback tier 3: `python -m p3_vlm_orchestrator.canned_run`

## Saturday exits (GT §5)

1. **AM:** one autonomous single-object cycle, photo → destination, no human input.  
2. **PM:** 6-object messy desk → clean unattended; induce a miss and watch retry heal; verify `keep`.

## Demo UI

Every photo + JSON decision under `runs/<timestamp>/`. Keep that screen visible for judges (GT §7).

## Layout

See [`STRUCTURE.md`](./STRUCTURE.md). Modules: `prompt`, `vlm_client`, `parse`, `policy`, `orchestrator`, `logging_ui`, `run_clean`, `canned_run`.
