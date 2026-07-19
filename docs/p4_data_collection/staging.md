# Staging for crumpled-paper episodes

1. Start each episode with arm at a consistent home / ready pose.
2. Place one crumpled ball in roughly the same zone region (±5 cm variation OK after first 20 eps).
3. Trash destination always the same taught container.
4. Reset: open gripper, clear any miss, re-crumple spare paper from the spare box.
5. Do not change camera mounts, lighting, or background mid-dataset (MolmoAct 2 views are frozen).
6. At grasp time, check both feeds — gripper should not fully occlude the paper in both views.
7. If an episode is garbage (collision, teleop glitch, occlusion), re-record — don't keep it.
