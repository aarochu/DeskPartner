# Desk Cleaner — Ground Truth / SOW

**Event:** Embodied Metal Hackathon · July 17–19, 2026  
**Entry track:** "Surprise us" (recycling challenge pivot kept as one-line pitch fallback)  
**Hardware:** reBot B601-DM follower + reBot 102 leader · **single-arm** · off the official SO-101 path  
**Repo:** https://github.com/aarochu/DeskPartner  
**Policy track:** **MolmoAct 2** fine-tune · **single-arm** config (station is normally bimanual / YAM-style; we run one arm)  
**Team:** 5 people (P1–P5); collapse to 4 covered in §8

Reference deck: [ReBot Arm Workshop](https://docs.google.com/presentation/d/1LXgehBvwPy5EhWO7aaYQffvmlN4eS1zFmJWYoDcTm-c/edit)

---

## 1. What we are building

**One sentence:** An overhead camera looks at a messy desk zone, a VLM decides what each item is and where it belongs, and the arm puts everything in its place (trash bin, pen cup, tray), re-checking after every pick until the desk is clean.

**Judging axes we optimize for:** ambition, accuracy, consistency, and how well the harness fits the skill.

| Layer | What it is | Who owns it |
|-------|------------|-------------|
| **Harness** | VLM planner + calibration + closed-loop orchestrator | P2 + P3 |
| **Skill A (must ship)** | Scripted IK pick/drop primitive | P1 |
| **Skill B (bonus)** | MolmoAct 2 single-arm fine-tune on venue demos (LoRA / action-expert-only) | P4 + P5 |

Sunday morning bake-off: 10 trials scripted vs 10 trials policy. Winner runs the demo.  
If B wins: *"fine-tuned a foundation VLA on data collected here, on hardware nobody else in the room ran."*

**Second act (near-free on QDD motors):** gravity-comp "weightless mode" + teach-by-grabbing replay. Flex: force-controlled wipe pass after zone is clear.

**Cut from MVP:** voice input, maze/writing, wrist camera.

---

## 2. System architecture

### 2.1 Coordinate frames (two calibrations)

1. **Pixel → plane.** Four 50 mm ArUco markers (4×4 dict, ids 0–3) at work-zone corners. One homography: camera pixel → mm on desk. Calibrate once Friday night.
2. **Plane → arm.** Jog tip to each marker center, record arm position, fit 2D transform (4 point pairs, least squares). Calibrate immediately after homography.

**Combined exit criterion:** click a pixel on screen → arm tip arrives within **5 mm**. Nothing else builds until this passes.

### 2.2 Perception (two layers)

- **Primary:** Photo → vision LLM (Claude or Gemini) → strict JSON: `{label, destination, bbox, desk_is_clean}`. Destinations constrained to `trash | pen_cup | tray | keep`. Local CV refines centroid inside each VLM box before arm conversion.
- **Fallback:** Classical CV vs empty-desk reference (contours + crude color rules). One flag. No network. Rehearse the switch once.

### 2.3 Decision

Open-vocab objects, constrained destinations. `"keep"` is never touched (judge phone beat). Policy is "VLM chose destination X" — that reasoning is visible on the laptop during demo.

### 2.4 Motion

- Home pose fully out of camera frame; photos **only** from home.
- Top-down pick only: hover → slow descend to per-type grasp height → close → lift → transit at safe height → release over taught destination.
- Destinations taught once by jogging, outside camera zone.
- No side grasps, no reorientation.

### 2.5 Closed loop

After every pick-and-drop → home → fresh photo. Still there → retry (max 2 per object, then skip). Dropped objects self-heal on next frame.

### 2.6 Handoff contracts

| From → To | Contract |
|-----------|----------|
| P2 → P3 | `pixel in → arm coordinates (mm) out` |
| P3 → P1 | `arm coordinates + category in → pick/drop happens` |
| P4 → P5 | Clean LeRobot dataset by Saturday midday |

If a contract is late, downstream works against faked inputs. Nobody blocks.

---

## 3. Physical setup

| Item | Spec |
|------|------|
| Camera | Overhead color innomaker (1080p wide or OV2719). **Not** OV9281 (mono). Mount 60–75 cm straight down. Direct USB, no hubs. Lock exposure + WB. |
| Track B cams | **Overhead + 45° side** (both color). MolmoAct 2 fine-tunes on whatever views the dataset defines — **lock camera setup before episode 1 and never move again.** Prefer views where the gripper does not block the object at grasp (known MolmoAct 2 weak spot). |
| Zone | ~40×30 cm, sized to **actual** B601 reach after jog-check. |
| Markers | Four 50 mm ArUco, ids 0–3. |
| Destinations | Trash bin, wide-mouth pen cup/jar, tray — outside zone. |
| Objects | Crumpled paper (hero), chunky markers/pens, foam blocks, light cup. Nothing flat/heavy/slippery. Spare staged set. |
| Host | Ubuntu 22.04 laptop. Follower `/dev/ttyACM0`, leader `/dev/ttyUSB0`. |
| Mounting | C-clamps on base, painter's tape, foam on gripper fingers. |

**Modal ($1k):** Track B MolmoAct 2 fine-tune GPU + inference serving + any harness hosting/evals. Visible in pitch.

---

## 4. Work split

| Person | Track | Owns |
|--------|-------|------|
| **P1** | A (must) | Zero cal, home pose, reach, pick/drop primitive, grasp heights, destinations, gravity-comp + teach-by-grabbing |
| **P2** | A (must) | Camera mount/exposure, ArUco homography, click-to-verify, plane→arm, empty-desk ref, CV fallback, centroid refine, 10-min recal procedure |
| **P3** | A (must) | VLM prompt/JSON parse, item→destination policy, closed-loop state machine, retry/skip, decision logging UI |
| **P4** | B (bonus) | Teleop practice, 50+ crumpled-paper episodes, locked dual-cam hygiene, **single-arm-only** LeRobot state/actions (no phantom second arm) |
| **P5** | B (bonus) | MolmoAct 2 single-arm fine-tune (LoRA or action-expert-only — not full FT), Modal train + serve, bake-off |

**Arm schedule:** A owns Friday night + Saturday morning. B gets Saturday midday data block. Training on Modal (no arm). B eval Saturday night + Sunday morning.

**Track B notes:** Lock overhead + 45° before episode 1. P4 verifies single-arm schema on Friday's throwaway (`verify_single_arm_dataset.py`). P5 confirms newt vs Ai2 MolmoAct 2 scripts + single-arm checkpoint with organizers Friday night; Saturday trains LoRA / action-expert-only only.

---

## 5. Timeline & exit criteria

### Friday night

1. Zero calibration (`2_zero_and_read.py`) before anything else  
2. Home pose + reach-check corners  
3. Tape zone to reach, markers, camera mount + lock  
4. Homography + registration → **click pixel → tip ≤ 5 mm**  
5. One hardcoded crumpled-paper pick-and-drop  
6. P4: teleop sanity + one throwaway episode + **single-arm dataset schema verify**  

**Sleep exit:** items 3 and 4 done.

### Saturday AM

- P1 hardens primitives + grasp heights  
- P3 VLM JSON clean on phone photos of real clutter  
- Wire VLM → orchestrator → arm  
- **Exit:** one autonomous single-object cycle, photo → destination, no human input  

### Saturday midday

- Track B: 50+ episodes  
- Track A: object set + photo-only tuning  

### Saturday PM

- Multi-object closed loop; induce failed grasp → retry heals  
- Verify `keep`  
- P5 kicks off MolmoAct 2 LoRA / action-expert fine-tune on Modal  
- **Exit:** 6-object messy desk → clean, unattended  

### Saturday night

- Edge cases, hardening, backup video #1  
- Wipe flex only if ahead  

### Sunday AM

- Bake-off decides scripted vs policy  
- Three full rehearsals of exact demo  
- Final backup video  
- **Code freeze** after rehearsal  

---

## 6. Risk register

| Risk | Mitigation |
|------|------------|
| Bad VLM coords | CV centroid refine inside box |
| Wrong destinations | Constrained list + few-shot in prompt |
| API down/slow | One-flag CV-only; rehearse switch |
| Grasp failures | Closed-loop retry, curated objects, foam fingers, top-down only |
| Bump camera/arm | C-clamps, taped mount, P2 10-min recal |
| Lighting shift | Locked exposure; re-shoot empty-desk ref |
| Track B eats schedule | Fixed arm blocks; B cuttable with zero guilt |
| Fine-tune doesn't converge | LoRA on ~50 eps of one task is realistic scope; one retry with fixed data Sat night, then cut B — no 4am hyperparam spelunking |
| Phantom bimanual channels | Station is normally two-arm; verify throwaway episode Friday has **only** active arm state/actions |
| Off SO-101 path | Mentors know SO-101, not damiao CAN; debug from our deck scripts; budget patience |
| Demo chaos | Fallback tiers, staged object box, one demo driver |

---

## 7. Demo run of show (~5 min)

1. Line: *"Your desk cleans itself. No teleop, no per-object script. It looks, decides where things go, and puts them there."*  
2. Dump staged mess → hit go. Narrate loop; laptop shows photo + VLM decisions.  
3. Judge puts something of theirs → tidy or `keep`.  
4. Optional: nudge mid-run → miss → retry heals.  
5. Second act: weightless mode → teach-by-grabbing if built.  
6. Closer: wipe pass if built.

**Fallback tiers (each is a working demo):**  
full VLM autonomy → CV-only autonomy → pre-staged canned run → backup video

---

## 8. Team of 4 collapse

- Merge **P4+P5** into one Track B owner.  
- Keep Track B **only** if Friday-night teleop feels smooth.  
- Otherwise cut B entirely; fourth person → demo polish, wipe flex, backup video.  
- Scripted pipeline alone is a complete competitive entry.

---

## 9. Venue check-in (30 min, Friday night)

- [ ] Off-menu arm eligible; "surprise us" is a real judged track  
- [ ] Booth: table size, power, lighting; second color innomaker available?  
- [ ] Modal team compute access working before Saturday  
- [ ] MolmoAct 2 specifics from organizers: which single-arm checkpoint/config, newt vs Ai2 scripts on Modal, working example of reBot data in expected LeRobot format  
- [ ] Jog real reach envelope, then tape zone  

---

## 10. Non-negotiable product truths

1. Photos only from home.  
2. Calibration exit is 5 mm click-to-tip — gates everything.  
3. Track B never blocks Track A.  
4. `keep` is never touched.  
5. Closed loop is the safety net; failures self-heal.  
6. Skill B = **MolmoAct 2 single-arm fine-tune** (LoRA / action-expert-only). Not full FT. Not bimanual YAM as deployed.  
7. Track B camera keys locked before episode 1 (overhead + 45°). Dataset has **no phantom second-arm channels**.  
8. Destinations taught once; zone sized to reach, not the reverse.  
9. Code freeze after Sunday AM rehearsal.  
