# Track B go / no-go gates

Read this under time pressure. Do not improvise.

---

## Saturday night — checkpoint eval

1. Run:
   ```bash
   python -m p5_training.eval_checkpoint --checkpoint PATH --trials 10
   ```
2. **Clearly bad** (rough guide: ≤3/10, or never grasps, or unsafe):
   - **One retry only** — allowed fixes:
     - more / cleaner demos (P4 re-record bad eps)
     - fixed data bug (phantom channels, wrong camera keys, truncated eps)
   - **NOT allowed:** hyperparameter spelunking, switching to full FT, overnight architecture changes
3. If the retry is also bad → **CUT TRACK B. No debate.** Demo runs scripted Skill A.
4. Log the decision in `runs/sat_night_gonogo.md` with scores.

## Sunday morning — bake-off

```bash
python -m p5_training.bakeoff --trials 10 --checkpoint PATH [--live]
```

- Same objects, same staging, scripted then policy.
- Read the printed line:
  ```
  VERDICT: WINNER=scripted|molmoact2 (...)
  ```
- **Winner runs the live demo.** Loser sits out.
- Tie → scripted (reliability).
- Policy losing is **fine** — still a talking point (“we fine-tuned a VLA on venue data”), not a failure.

## Hard rules

| Rule | Action |
|------|--------|
| Full fine-tune | Forbidden |
| Hung Modal job | Wall-clock timeout kills it; do not restart endlessly |
| Bad format data | Never train — fix P4 first |
| A slipping | Cut B immediately; put people on demo polish |

## Pitch if B wins

> Fine-tuned a foundation VLA on data collected here, on hardware nobody else in the room ran.
