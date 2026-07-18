# P3 structure

```
p3_vlm_orchestrator/
├── README.md
├── STRUCTURE.md
├── __init__.py
├── prompt.py              # system + few-shot JSON schema
├── vlm_client.py          # Claude / Gemini → PlanResult
├── parse.py               # robust JSON extraction
├── policy.py              # filter keep, attach grasp heights
├── orchestrator.py        # closed-loop state machine
├── logging_ui.py          # write runs/ trail
├── canned_run.py          # fallback tier 3
└── run_clean.py           # CLI entrypoint
```
