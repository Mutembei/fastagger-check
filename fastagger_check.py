# Copyright 2026 Fastagger Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fastagger first-pass check on a LeRobot v3 dataset (built for lerobot/droid_100).

python fastagger_check.py decode  <dataset> <out> <camera_key>   # decode one camera, cache motion/brightness
python fastagger_check.py analyze <dataset> <out>                # numeric + video + sync checks -> CSVs
python fastagger_check.py strips  <dataset> <out>                # example frames for flagged events
"""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, av, cv2

cmd, ds, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
info = json.loads((ds / "meta/info.json").read_text())
fps = info["fps"]
cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
short = lambda cam: cam.split(".")[-1].replace("_image", "").replace("_left", "")
vpath = lambda cam: next((ds / "videos" / cam).rglob("*.mp4"))
# Which state dims are the arm pose, which are wrapping angles, where the gripper is. DROID's layout is the default;
# fastagger_convert.py writes meta/fastagger.json for other robots.
cfg = json.loads((ds / "meta/fastagger.json").read_text()) if (ds / "meta/fastagger.json").exists() else {}
sdim = info["features"]["observation.state"]["shape"][0]; adim = info["features"]["action"]["shape"][0]
droid = sdim == 7 and not cfg
GRIP = cfg.get("gripper_dim", 6 if droid else None)
POSE = cfg.get("pose_dims", list(range(6)) if droid else [d for d in range(sdim) if d != GRIP])
ANG = [POSE.index(d) for d in cfg.get("angle_dims", [3, 4, 5] if droid else []) if d in POSE]
SPIKE = cfg.get("spike_dims", list(range(6)) if droid else [d for d in POSE if d < adim])
G_OPEN, G_CLOSED = cfg.get("gripper_open", 0.0), cfg.get("gripper_closed", 1.0)

if cmd == "decode":
    cam = sys.argv[4]
    t0 = time.time()
    c = av.open(str(vpath(cam))); st = c.streams.video[0]; st.thread_type = "AUTO"
    times, energy, bright, prev = [], [], [], None
    for fr in c.decode(st):
        small = cv2.resize(fr.to_ndarray(format="gray"), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        times.append(float(fr.pts * st.time_base)); bright.append(float(small.mean()))
        energy.append(float(np.abs(small - prev).mean()) if prev is not None else np.nan); prev = small
    c.close()
    np.savez(out / f"cache_{short(cam)}.npz", times=np.array(times), energy=np.array(energy), bright=np.array(bright), seconds=time.time() - t0)
    print(cam, len(times), "frames decoded in", round(time.time() - t0, 1), "s")
    sys.exit()

eps = pd.read_parquet(ds / "meta/episodes/chunk-000/file-000.parquet").sort_values("episode_index")

def runs(mask, min_len):
    res, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None: s = i
        elif not m and s is not None:
            if i - s >= min_len: res.append((s, i))
            s = None
    return res

if cmd == "analyze":
    t0 = time.time()
    data = pd.read_parquet(ds / "data/chunk-000/file-000.parquet")
    state = np.stack(data["observation.state"].to_numpy()).astype(float)
    action = np.stack(data["action"].to_numpy()).astype(float)
    ts, epi = data["timestamp"].to_numpy(float), data["episode_index"].to_numpy(int)
    vid = {cam: np.load(out / f"cache_{short(cam)}.npz") for cam in cams}
    decode_seconds = sum(float(v["seconds"]) for v in vid.values())
    # arm pose per episode with roll/pitch/yaw unwrapped (DROID Euler angles wrap at +/-pi)
    arm = {}
    for e in eps.episode_index:
        s = state[epi == e][:, POSE].copy()
        if ANG: s[:, ANG] = np.unwrap(s[:, ANG], axis=0)
        arm[e] = s
    scale = np.std(np.concatenate([np.diff(v, axis=0) for v in arm.values() if len(v) > 1]), axis=0) + 1e-9
    rows, events, hashes = [], [], {}
    def ev(e, code, a, b, detail): events.append(dict(episode=int(e), code=code, start_s=round(a, 2), end_s=round(b, 2), detail=detail))
    for _, er in eps.iterrows():
        e = int(er.episode_index); m = epi == e
        s, a, t = state[m], action[m], ts[m]; n = len(s)
        task = (er["tasks"][0] if len(er["tasks"]) else "").strip()
        flags = []
        if not task:
            flags.append("MISSING_INSTRUCTION"); ev(e, "MISSING_INSTRUCTION", 0, n / fps, "no language instruction")
        if n < 2 * fps:
            flags.append("TRUNCATED"); ev(e, "TRUNCATED", 0, n / fps, f"episode is only {n} frames ({n/fps:.1f}s)")
        dt = np.diff(t)
        gaps, back = int(np.sum(dt > 1.5 / fps)), int(np.sum(dt <= 0))
        if gaps or back: flags.append("TIMESTAMP_GAP"); ev(e, "TIMESTAMP_GAP", 0, n / fps, f"{gaps} gaps, {back} backwards steps")
        nans = int(np.isnan(s).sum() + np.isnan(a).sum())
        if nans: flags.append("MISSING_DATA"); ev(e, "MISSING_DATA", 0, n / fps, f"{nans} missing values")
        sp = np.r_[0, np.linalg.norm(np.diff(arm[e], axis=0) / scale, axis=1)] if n > 1 else np.zeros(n)
        moving = sp > 0.25
        idle = runs(~moving, int(3 * fps))
        idle_s = float((~moving).sum()) / fps
        for a_, b in idle:
            ev(e, "LONG_IDLE", a_ / fps, b / fps, f"{(b-a_)/fps:.1f}s with no arm motion")
        if idle: flags.append("LONG_IDLE")
        # action spikes: a one-step blip of size >0.5 (actions are normalized to [-1, 1]) that reverses at once
        da = np.diff(a[:, SPIKE], axis=0); spikes = 0
        for i in range(len(da) - 1):
            for jj, j in enumerate(SPIKE):
                if abs(da[i, jj]) > 0.5 and abs(da[i + 1, jj]) > 0.5 and np.sign(da[i, jj]) != np.sign(da[i + 1, jj]):
                    spikes += 1; ev(e, "ACTION_SPIKE", (i + 1) / fps, (i + 2) / fps, f"action dim {j}: {a[i,j]:.2f} -> {a[i+1,j]:.2f} -> {a[i+2,j]:.2f}")
        if spikes: flags.append("ACTION_SPIKE")
        # gripper: holding (partly closed, steady) then snapping fully shut suggests a lost object
        g = (s[:, GRIP] - G_OPEN) / (G_CLOSED - G_OPEN) if GRIP is not None else np.zeros(n); drop_at = None
        for i in range(8, n - 8 if GRIP is not None else 0):
            hold = g[i - 8:i]
            if 0.2 < np.median(hold) < 0.9 and np.ptp(hold) < 0.05 and g[i + 8] > 0.97 and g[i + 8] - np.median(hold) > 0.08:
                drop_at = i; break
        if drop_at is not None:
            flags.append("POSSIBLE_DROP"); ev(e, "POSSIBLE_DROP", drop_at / fps, (drop_at + 8) / fps, f"gripper {np.median(g[drop_at-8:drop_at]):.2f} -> {g[drop_at+8]:.2f} while holding")
        closes = len(runs(g > 0.5, 3)) if GRIP is not None else 0
        if closes >= 3:
            flags.append("MANY_GRASPS"); ev(e, "MANY_GRASPS", 0, n / fps, f"gripper closed {closes} times")
        h = hashlib.sha1(np.round(s, 4).tobytes()).hexdigest(); dup_of = hashes.get(h); hashes.setdefault(h, e)
        if dup_of is not None: flags.append("DUPLICATE"); ev(e, "DUPLICATE", 0, n / fps, f"same trajectory as episode {dup_of}")
        rec = dict(episode=e, task=task, frames=n, seconds=round(n / fps, 2), idle_seconds=round(idle_s, 1),
                   idle_fraction=round(idle_s / max(n / fps, 1e-9), 2), action_spikes=spikes, grasps=closes,
                   timestamp_gaps=gaps, missing_values=nans, duplicate_of=dup_of if dup_of is not None else "")
        for cam in cams:
            times, energy, bright = vid[cam]["times"], vid[cam]["energy"], vid[cam]["bright"]
            f0, f1 = float(er[f"videos/{cam}/from_timestamp"]), float(er[f"videos/{cam}/to_timestamp"])
            sel = (times >= f0 - 0.5 / fps) & (times < f1 - 0.5 / fps)
            en, br = energy[sel], bright[sel]; k = min(len(en), n); sh = short(cam)
            rec[f"{sh}_frames"] = int(sel.sum())
            if abs(int(sel.sum()) - n) > 1:
                flags.append("DROPPED_FRAMES"); ev(e, "DROPPED_FRAMES", 0, n / fps, f"{sh}: {int(sel.sum())} video frames for {n} steps")
            frozen = 0
            if k > 10:
                en2 = en[:k].copy(); en2[0] = 99
                for a_, b in runs((en2 < 0.15) & moving[:k], 8):
                    frozen += 1; ev(e, "FROZEN_VIDEO", a_ / fps, b / fps, f"{sh}: {(b-a_)/fps:.1f}s of unchanged frames while the arm moves")
            rec[f"{sh}_frozen_runs"] = frozen
            if frozen: flags.append("FROZEN_VIDEO")
            rec[f"{sh}_dark_frames"] = int(np.sum(br[:k] < 8))
            if k > 45 and np.std(sp[1:k]) > 0:
                x = (sp[1:k] - sp[1:k].mean()) / sp[1:k].std()
                y = np.nan_to_num(en[1:k]); y = (y - y.mean()) / (y.std() + 1e-9)
                best, bl = -9.0, 0
                for lag in range(-6, 7):
                    xx, yy = (x[:len(x) - lag], y[lag:]) if lag >= 0 else (x[-lag:], y[:len(y) + lag])
                    if len(xx) > 10:
                        r = float(np.corrcoef(xx, yy)[0, 1])
                        if np.isfinite(r) and r > best: best, bl = r, lag
                rec[f"{sh}_lag_frames"], rec[f"{sh}_lag_corr"] = bl, round(best, 2)
                if sh.startswith("wrist") and best >= 0.4 and abs(bl) >= 2:
                    flags.append("SYNC_OFFSET"); ev(e, "SYNC_OFFSET", 0, n / fps, f"wrist video lags arm motion by {bl} frames ({bl*1000/fps:.0f} ms), r={best:.2f}")
        rec["flags"] = ";".join(sorted(set(flags)))
        rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(out / "per_episode.csv", index=False)
    evs = pd.DataFrame(events); evs.to_csv(out / "events.csv", index=False)
    codes = sorted(set(evs.code)) if len(evs) else []
    analysis_s = time.time() - t0
    summary = dict(episodes=len(df), frames=int(df.frames.sum()), video_minutes=round(df.frames.sum() / fps / 60, 1),
                   seconds_mean=round(df.seconds.mean(), 1), seconds_median=round(df.seconds.median(), 1),
                   seconds_p10=round(df.seconds.quantile(0.1), 1), seconds_p90=round(df.seconds.quantile(0.9), 1),
                   seconds_min=round(df.seconds.min(), 1), seconds_max=round(df.seconds.max(), 1),
                   idle_fraction_median=round(df.idle_fraction.median(), 2),
                   episodes_with_any_flag=int((df["flags"] != "").sum()),
                   episodes_by_code={c: int(df["flags"].str.split(";").apply(lambda L: c in L).sum()) for c in codes},
                   events_by_code={c: int((evs.code == c).sum()) for c in codes},
                   decode_seconds=round(decode_seconds, 1), analysis_seconds=round(analysis_s, 1),
                   compute_seconds_per_episode=round((decode_seconds + analysis_s) / len(df), 2))
    (out / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))

if cmd == "strips":
    evs = pd.read_csv(out / "events.csv"); (out / "frames").mkdir(exist_ok=True)
    pick = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    def grab(cam, t_list):
        c = av.open(str(vpath(cam))); st = c.streams.video[0]; got_all = []
        for t in t_list:
            c.seek(int(max(0.0, t - 0.1) / st.time_base), stream=st, backward=True, any_frame=False); got = None
            for fr in c.decode(st):
                if float(fr.pts * st.time_base) >= t - 0.5 / fps: got = fr.to_ndarray(format="bgr24"); break
            got_all.append(got if got is not None else np.zeros((180, 320, 3), np.uint8))
        c.close(); return got_all
    done = {}
    for _, r in evs.iterrows():
        if pick and f"{r.code}:{int(r.episode)}" not in pick and r.code not in pick: continue
        if not pick and done.get(r.code, 0) >= 2: continue
        er = eps[eps.episode_index == r.episode].iloc[0]; n = int(er["length"])
        c0 = int(r.start_s * fps); ids = sorted(set(min(n - 1, max(0, c0 + d)) for d in (-10, -5, 0, 5, 10, 15)))
        rows = []
        for cam in cams:
            f0 = float(er[f"videos/{cam}/from_timestamp"]); imgs = grab(cam, [f0 + i / fps for i in ids])
            for img, i in zip(imgs, ids):
                cv2.putText(img, f"{short(cam)} t={i/fps:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.hstack(imgs))
        img = np.vstack(rows)
        bar = np.zeros((26, img.shape[1], 3), np.uint8)
        cv2.putText(bar, f"episode {int(r.episode)}  {r.code}  {r.detail}"[:150], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / "frames" / f"{r.code}_ep{int(r.episode):03d}_t{r.start_s:.1f}.jpg"), np.vstack([bar, img]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        done[r.code] = done.get(r.code, 0) + 1
    print("strips written:", done)
