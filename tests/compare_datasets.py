# Copyright 2026 Fastagger Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare a converted LeRobot v3 dataset with its source: state and action value for value, and each camera's
frames by PSNR at frame offsets -1, 0 and +1 (the best match should be 0: no frame shift).

python tests/compare_datasets.py <source dataset> <converted dataset> [first source episode, default 0]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, av, cv2

src, new, off = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0

def load(ds):
    return (pd.read_parquet(ds / "data/chunk-000/file-000.parquet"),
            pd.read_parquet(ds / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index"),
            json.loads((ds / "meta/info.json").read_text()))

def frames(ds, cam, f0, k, fps):
    c = av.open(str(next((ds / "videos" / cam).rglob("*.mp4")))); s = c.streams.video[0]; out = []
    c.seek(int(max(0, f0 - 1) / s.time_base), stream=s, backward=True)
    for fr in c.decode(s):
        if float(fr.pts * s.time_base) < f0 - 0.5 / fps: continue
        out.append(cv2.cvtColor(fr.to_ndarray(format="yuv420p"), cv2.COLOR_YUV2RGB_I420).astype(np.float32))
        if len(out) == k: break
    c.close(); return out

psnr = lambda x, y: 10 * np.log10(255 ** 2 / max(float(np.mean((x - y) ** 2)), 1e-9))
ds_, es, info = load(src); dn, en, _ = load(new); fps = info["fps"]
worst, bad = 0.0, 0
for e in en.index:
    for col in ("observation.state", "action"):
        a = np.stack(ds_[ds_.episode_index == e + off][col]); b = np.stack(dn[dn.episode_index == e][col])
        if a.shape != b.shape: print(f"episode {e}: {col} has shape {b.shape}, source {a.shape}"); bad += 1; continue
        worst = max(worst, float(np.abs(a - b).max()))
print(f"state and action: largest difference {worst} over {len(en)} episodes" + (f", {bad} shape mismatches" if bad else ""))
cams = [k for k, v in info["features"].items() if v.get("dtype") == "video" and f"videos/{k}/from_timestamp" in en.columns]
shifted, scores = 0, []
for e in list(en.index)[:: max(1, len(en) // 5)]:
    for cam in cams:
        A = frames(src, cam, float(es.loc[e + off, f"videos/{cam}/from_timestamp"]), 40, fps)
        B = frames(new, cam, float(en.loc[e, f"videos/{cam}/from_timestamp"]), 40, fps)
        res = {d: np.mean([psnr(A[i], B[i + d]) for i in range(2, min(len(A), len(B)) - 2)]) for d in (-1, 0, 1)}
        scores.append(res[0]); shifted += max(res, key=res.get) != 0
        print(f"episode {e:3d} {cam.split('.')[-1]:24} PSNR at offset -1/0/+1: " + " / ".join(f"{res[d]:.1f}" for d in (-1, 0, 1)))
print(f"video: PSNR {min(scores):.1f} to {max(scores):.1f} dB at offset 0; {shifted} camera-episodes matched best at another offset")
