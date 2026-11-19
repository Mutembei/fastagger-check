# Copyright 2026 Fastagger Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn per_episode.csv into one verdict per episode. The most serious bucket wins.

python verdicts.py <out>    # reads <out>/per_episode.csv, writes <out>/episode_verdicts.csv
"""
import sys
from pathlib import Path
import pandas as pd

BUCKETS = [  # most serious first
    ("Re-record", {"TRUNCATED"}),
    ("For a person", {"POSSIBLE_DROP", "MANY_GRASPS", "SYNC_OFFSET", "FROZEN_VIDEO", "DROPPED_FRAMES", "DUPLICATE"}),
    ("Software fix", {"MISSING_INSTRUCTION", "LONG_IDLE", "ACTION_SPIKE", "TIMESTAMP_GAP", "MISSING_DATA"}),
]

def verdict(flags):
    found = set(f for f in str(flags).split(";") if f and f != "nan")
    for name, codes in BUCKETS:
        if found & codes:
            return name
    return "Clean"

out = Path(sys.argv[1])
df = pd.read_csv(out / "per_episode.csv", keep_default_na=False)
df["verdict"] = df["flags"].apply(verdict)
df[["episode", "task", "seconds", "verdict", "flags"]].to_csv(out / "episode_verdicts.csv", index=False)
print(df["verdict"].value_counts().to_string())
