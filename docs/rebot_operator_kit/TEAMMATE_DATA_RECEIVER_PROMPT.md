# Prompt for the teammate receiving ReBot data

Copy everything below into the teammate's Codex task.

---

You are setting up the receiving side of our DeskPartner ReBot LeRobot dataset
sharing workflow. Work carefully and leave the machine ready for repeatable,
incremental updates.

Source repository: `https://github.com/aarochu/DeskPartner`

Expected initial dataset repository:
`<SENDER_WILL_PROVIDE_OWNER_OR_ORG>/rebot-can-sort-stage1-v1-smoke`

Dataset contract:

- LeRobot v3, success-only demonstrations.
- Task: `Pick up one can and place it in the taped sorting zone`.
- 30 FPS.
- `observation.images.front`: Logitech overhead, 640x480.
- `observation.images.side`: Innomaker wrist/claw, 1280x720.
- Seven follower-space state/action values in this order: shoulder pan,
  shoulder lift, elbow flex, wrist flex, wrist yaw, wrist roll, gripper.

Do the following:

1. Clone or fast-forward `aarochu/DeskPartner` without overwriting unrelated
   local work. Read `rebot_operator_kit/AGENTS.md`, `README.md`, and
   `TRAINING_GUIDE.md` completely.
2. Set up the pinned ReBot/LeRobot runtime using the repository instructions.
   Never request or copy our robot calibration secrets unless your machine
   actually controls the same physical arm; receiving/training does not need
   serial-port ownership.
3. Install/use the isolated current CLI and log into Hugging Face through
   browser/device OAuth:

   ```bash
   ./10_hf_oauth_login.command
   ```

   Never paste a token into chat, Git, a script, or a committed config file.
4. Download the exact dataset revision I provide, not merely whatever `main`
   points to:

   ```bash
   cd rebot_operator_kit
   ./09_receive_dataset.command hub \
     <OWNER_OR_ORG>/rebot-can-sort-stage1-v1-smoke \
     "$HOME/rebot-training/data/rebot-can-sort-stage1-v1-smoke" \
     <IMMUTABLE_REVISION>
   ```

5. Confirm the command verifies every SHA-256 checksum and completes the full
   LeRobot validation. Do not train from a partial download or a moving folder
   while it is being updated.
6. Reply with: machine/host alias, destination path, Hugging Face username,
   whether you can access the private dataset, exact immutable revision,
   episode count, frame count, checksum result, validation result, and any
   blocker. Do not send credentials or private tokens.
7. For later batches, rerun the same receive command with the newly provided
   immutable revision. Hugging Face will reuse its cache and download only the
   required changed content. Keep the revision used by every training run in
   its run manifest.

If we choose direct machine-to-machine transfer instead, send me a safe SSH
destination in the form `user@host:/absolute/path`, confirm that key-based SSH
access is working, receive the folder, then run:

```bash
./09_receive_dataset.command verify /absolute/path/rebot-can-sort-stage1-v1-smoke
```

Do not put MP4, Parquet, Rerun, model, token, cache, or calibration files into
the normal DeskPartner Git history. Report evidence, not assumptions.

---
