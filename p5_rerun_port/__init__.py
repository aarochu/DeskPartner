"""reBot end-to-end Rerun port (SO-101 pipeline shape, non-SO-101 hardware).

Priority loop:
  log_rebot → record_episode (.rrd) → query_dataset → export_lerobot → replay_episode
"""

__version__ = "0.1.0"
