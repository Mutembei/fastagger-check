# Copyright 2026 Fastagger Inc.
# SPDX-License-Identifier: Apache-2.0
"""Convert robot recordings to LeRobot v3, and LeRobot v3 to MCAP for Foxglove.

python fastagger_convert.py inspect   <file or folder>
python fastagger_convert.py to-lerobot <inputs...> --out <dataset> --fps 15 [stream options]
python fastagger_convert.py to-mcap   <dataset> --out <folder> [--episodes 0-19] [--flags out/events.csv]

Inputs for to-lerobot (one file, or one ROS 2 bag folder, is one episode):
  .hdf5 / .h5   ALOHA layout (observations/qpos, action, observations/images/<cam>), robomimic layout
                (data/demo_N/obs, data/demo_N/actions; every demo is an episode), or any layout with --state/--action/--camera keys
  .bag          ROS 1 bag
  ROS 2 bag     a folder with metadata.yaml, or a .db3 file inside one
  .mcap         ROS 2 (CDR), Protobuf (including Foxglove schemas) or JSON messages

Streams are named the same way for every format:
  --state  /joint_states[:position]      a topic (bags, MCAP) or dataset path (HDF5), optionally :field.
                                         Repeat to join several into one state vector, in order.
  --action /arm/command[:data]           same, for the action vector (optional)
  --camera wrist=/wrist/image_raw        camera name = topic or dataset path. Repeat for each camera.
  --task "Put the cup in the sink"       language instruction, or --task-topic /instruction
"""
import argparse, json, sys
from fractions import Fraction
from pathlib import Path
import numpy as np, pandas as pd, av, cv2

# ------------------------------------------------------------------ LeRobot v3 writer

def _vec_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1: x = x[:, None]
    return dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=x.std(0), count=np.array([len(x)]))

def _merge_stats(parts):
    n = np.array([p["count"][0] for p in parts], dtype=np.float64)
    mean = sum(p["mean"] * k for p, k in zip(parts, n)) / n.sum()
    var = sum((p["std"] ** 2 + (p["mean"] - mean) ** 2) * k for p, k in zip(parts, n)) / n.sum()
    return dict(min=np.min([p["min"] for p in parts], 0), max=np.max([p["max"] for p in parts], 0),
                mean=mean, std=np.sqrt(var), count=np.array([int(n.sum())]))

def _tolist(d): return {k: np.asarray(v).tolist() for k, v in d.items()}

# swscale's default RGB<->YUV conversions round down, which darkens video by about one level per pass and moves
# the check's brightness and motion numbers. OpenCV's BT.601 limited-range conversion keeps the luma exact.
def _rgb_to_frame(rgb):
    return av.VideoFrame.from_ndarray(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV_I420), format="yuv420p")

def _frame_to_bgr(fr):
    if fr.format.name == "yuv420p" and fr.color_range != 2 and fr.width % 2 == 0 and fr.height % 2 == 0:
        return cv2.cvtColor(fr.to_ndarray(format="yuv420p"), cv2.COLOR_YUV2BGR_I420)
    return fr.to_ndarray(format="bgr24")


class LeRobotWriter:
    """Writes one LeRobot v3 dataset: a single data file and a single video file per camera."""

    def __init__(self, out, fps, robot_type="unknown", codec="libx264", crf=18):
        self.out, self.fps, self.robot_type, self.codec, self.crf = Path(out), fps, robot_type, codec, crf
        if (self.out / "meta/info.json").exists():
            sys.exit(f"{self.out} already holds a dataset; choose a new --out folder")
        self.rows, self.episodes, self.tasks, self.video = [], [], {}, {}
        self.state_names = self.action_names = None
        self.ep_stats = []

    def _open_video(self, cam, h, w):
        h2, w2 = h - h % 2, w - w % 2
        p = self.out / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"; p.parent.mkdir(parents=True, exist_ok=True)
        c = av.open(str(p), "w")
        s = c.add_stream(self.codec, rate=self.fps); s.width, s.height, s.pix_fmt = w2, h2, "yuv420p"
        s.options = {"crf": str(self.crf), "g": "2"}
        self.video[cam] = dict(c=c, s=s, h=h2, w=w2, n=0)

    def add_episode(self, state, action, cameras, task="", state_names=None, action_names=None):
        """state, action: (n, d) arrays. cameras: {name: callable returning an iterator of n RGB uint8 frames}."""
        state = np.asarray(state, np.float32); n = len(state)
        if n == 0: print("  skipped an empty episode"); return
        action = np.asarray(action, np.float32) if action is not None else state.copy()
        if len(action) != n: sys.exit(f"state has {n} steps but action has {len(action)}")
        if self.state_names is None:
            self.state_names = list(state_names or [f"state_{i}" for i in range(state.shape[1])])
            self.action_names = list(action_names or [f"action_{i}" for i in range(action.shape[1])])
            self.state_dim, self.action_dim, self.cams = state.shape[1], action.shape[1], sorted(cameras)
        if state.shape[1] != self.state_dim or action.shape[1] != self.action_dim:
            sys.exit(f"episode {len(self.episodes)} has state/action sizes {state.shape[1]}/{action.shape[1]}, "
                     f"earlier episodes had {self.state_dim}/{self.action_dim}")
        if sorted(cameras) != self.cams: sys.exit(f"episode {len(self.episodes)} has cameras {sorted(cameras)}, expected {self.cams}")
        e, first = len(self.episodes), len(self.rows)
        ti = self.tasks.setdefault(task, len(self.tasks))
        ts = np.arange(n, dtype=np.float32) / self.fps
        for i in range(n):
            self.rows.append({"observation.state": state[i], "action": action[i], "timestamp": float(ts[i]),
                              "episode_index": e, "frame_index": i, "next.done": i == n - 1,
                              "index": first + i, "task_index": ti})
        rec = {"episode_index": e, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": first, "dataset_to_index": first + n}
        stats = {"observation.state": _vec_stats(state), "action": _vec_stats(action), "timestamp": _vec_stats(ts),
                 "episode_index": _vec_stats(np.full(n, e)), "frame_index": _vec_stats(np.arange(n)),
                 "index": _vec_stats(np.arange(first, first + n)), "task_index": _vec_stats(np.full(n, ti)),
                 "next.done": _vec_stats(np.r_[np.zeros(n - 1), 1.0])}
        for cam in self.cams:
            got, samples = 0, []
            for frame in cameras[cam]():
                if got >= n: break
                frame = np.asarray(frame)
                if frame.ndim == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                if cam not in self.video: self._open_video(cam, frame.shape[0], frame.shape[1])
                v = self.video[cam]
                if frame.shape[:2] != (v["h"], v["w"]): frame = cv2.resize(frame, (v["w"], v["h"]), interpolation=cv2.INTER_AREA)
                vf = _rgb_to_frame(frame[..., :3])
                vf.pts = v["n"]; vf.time_base = Fraction(1, self.fps)
                for pkt in v["s"].encode(vf): v["c"].mux(pkt)
                if got % 10 == 0: samples.append(cv2.resize(frame[..., :3], (64, 36)).reshape(-1, 3) / 255.0)
                v["n"] += 1; got += 1
            if got != n: sys.exit(f"episode {e}, camera {cam}: {got} frames for {n} steps")
            v = self.video[cam]; key = f"observation.images.{cam}"
            rec.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                        f"videos/{key}/from_timestamp": (v["n"] - n) / self.fps, f"videos/{key}/to_timestamp": v["n"] / self.fps})
            px = np.concatenate(samples); st = _vec_stats(px)
            stats[key] = {k: (val.reshape(3, 1, 1) if k != "count" else np.array([n])) for k, val in st.items()}
        rec["tasks"] = [task]; rec["length"] = n
        self.episodes.append(rec); self.ep_stats.append(stats)
        print(f"  episode {e}: {n} steps, {len(self.cams)} cameras, task: {task or '(none)'}")

    def close(self, extra_meta=None):
        if not self.episodes: sys.exit("no episodes were written")
        for v in self.video.values():
            for pkt in v["s"].encode(None): v["c"].mux(pkt)
            v["c"].close()
        (self.out / "data/chunk-000").mkdir(parents=True, exist_ok=True); (self.out / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)
        df["observation.state"] = [r.tolist() for r in df["observation.state"]]; df["action"] = [r.tolist() for r in df["action"]]
        df.to_parquet(self.out / "data/chunk-000/file-000.parquet", index=False)
        eps = pd.DataFrame(self.episodes)
        for feat in self.ep_stats[0]:
            for k in ("min", "max", "mean", "std", "count"):
                eps[f"stats/{feat}/{k}"] = [s[feat][k].tolist() for s in self.ep_stats]
        eps["meta/episodes/chunk_index"] = 0; eps["meta/episodes/file_index"] = 0
        eps.to_parquet(self.out / "meta/episodes/chunk-000/file-000.parquet", index=False)
        pd.DataFrame({"task_index": list(self.tasks.values())}, index=pd.Index(list(self.tasks), name=None)).to_parquet(self.out / "meta/tasks.parquet")
        stats = {f: _tolist(_merge_stats([s[f] for s in self.ep_stats])) for f in self.ep_stats[0]}
        (self.out / "meta/stats.json").write_text(json.dumps(stats, indent=2))
        vec = lambda dt, names: {"dtype": dt, "shape": [len(names)], "names": {"motors": names}, "fps": float(self.fps)}
        one = lambda dt: {"dtype": dt, "shape": [1], "names": None, "fps": float(self.fps)}
        feats = {}
        for cam in self.cams:
            v = self.video[cam]
            feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": [v["h"], v["w"], 3], "names": ["height", "width", "channel"],
                "video_info": {"video.fps": float(self.fps), "video.codec": "h264" if self.codec == "libx264" else "av1",
                               "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
        feats["observation.state"] = vec("float32", self.state_names); feats["action"] = vec("float32", self.action_names)
        for k, dt in (("timestamp", "float32"), ("episode_index", "int64"), ("frame_index", "int64"),
                      ("next.done", "bool"), ("index", "int64"), ("task_index", "int64")):
            feats[k] = one(dt)
        info = {"codebase_version": "v3.0", "robot_type": self.robot_type, "total_episodes": len(self.episodes),
                "total_frames": len(self.rows), "total_tasks": len(self.tasks), "chunks_size": 1000, "fps": self.fps,
                "splits": {"train": f"0:{len(self.episodes)}"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": feats, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500}
        (self.out / "meta/info.json").write_text(json.dumps(info, indent=2))
        if extra_meta: (self.out / "meta/fastagger.json").write_text(json.dumps(extra_meta, indent=2))
        print(f"wrote {len(self.episodes)} episodes, {len(self.rows)} steps to {self.out}")

# ------------------------------------------------------------------ message helpers (bags and MCAP)

def _norm_type(t): return t.replace("/msg/", "/").replace("::", "/").replace(".", "/")

def _field(msg, path):
    for part in [p for p in path.split(".") if p]:
        msg = msg[part] if isinstance(msg, dict) else getattr(msg, part)
    return msg

def _stamp(msg):
    try:
        h = _field(msg, "header.stamp"); return int(getattr(h, "sec", 0)) * 10**9 + int(getattr(h, "nanosec", getattr(h, "nsec", 0)))
    except Exception:
        try: t = _field(msg, "timestamp"); return int(t.sec) * 10**9 + int(t.nsec)
        except Exception: return None

def _vector(msg, mtype, field):
    """A numeric vector, plus its element names when the message has them."""
    t = _norm_type(mtype)
    if field:
        v = _field(msg, field)
        if isinstance(v, (int, float, bool)): return np.array([float(v)]), None
        return np.asarray(list(v), dtype=np.float64), None
    if t.endswith("JointState"): return np.asarray(list(msg.position), np.float64), [str(x) for x in msg.name]
    if t.endswith("JointStates"):
        return np.array([j.position for j in msg.joints], np.float64), [j.name for j in msg.joints]
    if "MultiArray" in t: return np.asarray(list(msg.data), np.float64), None
    if t.endswith("PoseStamped") or t.endswith("Pose"):
        p = msg.pose if hasattr(msg, "pose") else msg
        return np.array([p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]), \
               ["x", "y", "z", "qx", "qy", "qz", "qw"]
    if t.endswith("Float64") or t.endswith("Float32") or t.endswith("Bool"): return np.array([float(msg.data)]), None
    if isinstance(msg, dict):
        for k in ("position", "data", "state", "action", "values"):
            if k in msg: return np.asarray(msg[k], np.float64), msg.get("names")
    sys.exit(f"don't know how to read a vector from {mtype}; name the field, e.g. /topic:position")

def _image_record(msg, mtype, field):
    """Keep images compressed in memory; decode only when writing."""
    if field: msg = _field(msg, field)
    t = _norm_type(mtype)
    if "Compressed" in t or (isinstance(msg, dict) and "format" in msg):
        data = msg["data"] if isinstance(msg, dict) else msg.data
        if isinstance(data, str): import base64; data = base64.b64decode(data)
        return ("enc", bytes(data))
    if t.endswith("Image") or t.endswith("RawImage"):
        enc = str(msg.encoding).lower(); h, w = int(msg.height), int(msg.width)
        a = np.frombuffer(bytes(msg.data), np.uint8)
        step = int(getattr(msg, "step", 0) or 0)
        if enc in ("rgb8", "bgr8"): a = a.reshape(h, step or w * 3)[:, :w * 3].reshape(h, w, 3)
        elif enc in ("rgba8", "bgra8"): a = a.reshape(h, step or w * 4)[:, :w * 4].reshape(h, w, 4)[..., :3]
        elif enc in ("mono8", "8uc1"): a = cv2.cvtColor(a.reshape(h, step or w)[:, :w], cv2.COLOR_GRAY2RGB)
        else: sys.exit(f"image encoding {enc} is not supported yet")
        if enc.startswith("bgr"): a = a[..., ::-1]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR))
        return ("enc", buf.tobytes())
    sys.exit(f"don't know how to read an image from {mtype}")

def _decode(rec):
    img = cv2.imdecode(np.frombuffer(rec[1], np.uint8), cv2.IMREAD_COLOR)
    if img is None: sys.exit("an image could not be decoded")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _iter_messages(path):
    """Yield (topic, type name, log time ns, decoded message) from a bag or MCAP file."""
    path = Path(path)
    if path.suffix == ".mcap" and not (path.parent / "metadata.yaml").exists():
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory as Ros2
        from mcap_protobuf.decoder import DecoderFactory as Proto
        class Json:
            def decoder_for(self, enc, schema): return (lambda b: json.loads(b)) if enc == "json" else None
        with open(path, "rb") as f:
            for schema, ch, msg, dec in make_reader(f, decoder_factories=[Ros2(), Proto(), Json()]).iter_decoded_messages():
                yield ch.topic, (schema.name if schema else "json"), msg.log_time, dec
        return
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    bag = path.parent if path.suffix in (".db3", ".mcap") else path
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        for conn, t, raw in r.messages():
            yield conn.topic, conn.msgtype, t, r.deserialize(raw, conn.msgtype)

def _split_spec(spec):
    topic, _, field = spec.partition(":")
    return topic, field

# ------------------------------------------------------------------ readers

def read_log(path, a):
    """One bag or MCAP file -> one episode, resampled to a fixed rate."""
    want = {}
    for i, s in enumerate(a.state): want.setdefault(_split_spec(s)[0], []).append(("state", i, _split_spec(s)[1]))
    for i, s in enumerate(a.action or []): want.setdefault(_split_spec(s)[0], []).append(("action", i, _split_spec(s)[1]))
    cams = dict(c.split("=", 1) for c in a.camera)
    for name, s in cams.items(): want.setdefault(_split_spec(s)[0], []).append(("camera", name, _split_spec(s)[1]))
    if a.task_topic: want.setdefault(a.task_topic, []).append(("task", 0, ""))
    got = {}; names = {}; task = a.task or ""
    for topic, mtype, t, msg in _iter_messages(path):
        if topic not in want: continue
        if a.stamp == "header": t = _stamp(msg) or t
        for kind, key, field in want[topic]:
            if kind == "task":
                d = msg if isinstance(msg, dict) else {k: getattr(msg, k) for k in ("data", "message") if hasattr(msg, k)}
                task = str(d.get("data", d.get("message", ""))); continue
            if kind == "camera": got.setdefault((kind, key), []).append((t, _image_record(msg, mtype, field)))
            else:
                v, nm = _vector(msg, mtype, field); got.setdefault((kind, key), []).append((t, v)); names.setdefault((kind, key), nm)
    need = [("state", i) for i in range(len(a.state))] + [("action", i) for i in range(len(a.action or []))] + [("camera", c) for c in cams]
    missing = [k for k in need if k not in got]
    if missing: sys.exit(f"{path}: no messages for {', '.join(f'{k[0]} {k[1]}' for k in missing)}. Run 'inspect' to see the topics.")
    for k in got: got[k].sort(key=lambda x: x[0])
    t0 = max(got[k][0][0] for k in need); t1 = min(got[k][-1][0] for k in need)
    n = int(round((t1 - t0) * a.fps / 1e9)) + 1
    grid = t0 + np.round(np.arange(n) * 1e9 / a.fps).astype(np.int64)
    report = {"file": str(path), "steps": n}
    def pick(k, nearest=True):
        """For each step, the message nearest in time. The report says how far off the matches were."""
        ts = np.array([x[0] for x in got[k]], np.int64)
        idx = np.clip(np.searchsorted(ts, grid, "left"), 0, len(ts) - 1)
        prev = np.maximum(idx - 1, 0); idx = np.where(np.abs(ts[prev] - grid) <= np.abs(ts[idx] - grid), prev, idx)
        off = np.abs(grid - ts[idx]) / 1e9
        label = f"{k[0]}_{k[1]}"
        report[f"{label}_messages"] = len(ts)
        report[f"{label}_rate_hz"] = round((len(ts) - 1) / max((ts[-1] - ts[0]) / 1e9, 1e-9), 1)
        report[f"{label}_max_gap_s"] = round(float(np.max(np.diff(ts)) / 1e9) if len(ts) > 1 else 0.0, 3)
        report[f"{label}_reused_steps"] = int(np.sum(np.diff(idx) == 0))
        report[f"{label}_stale_steps"] = int(np.sum(off > 1.0 / a.fps))
        return idx
    def stack(kind, count):
        if not count: return None, None
        parts, nm = [], []
        for i in range(count):
            idx = pick((kind, i)); arr = np.stack([got[(kind, i)][j][1] for j in idx]); parts.append(arr)
            nm += names.get((kind, i)) or [f"{kind}_{len(nm) + d}" for d in range(arr.shape[1])]
        return np.concatenate(parts, 1), nm
    state, snames = stack("state", len(a.state)); action, anames = stack("action", len(a.action or []))
    cam_fns = {}
    for c in cams:
        idx = pick(("camera", c)); recs = got[("camera", c)]
        cam_fns[c] = (lambda recs=recs, idx=idx: (_decode(recs[j][1]) for j in idx))
    return [dict(state=state, action=action, cameras=cam_fns, task=task, state_names=snames, action_names=anames, report=report)]

def read_hdf5(path, a):
    import h5py
    f = h5py.File(path, "r")
    def frames(ds, bgr):
        def it():
            for i in range(ds.shape[0]):
                x = ds[i]
                if x.ndim == 1: x = cv2.cvtColor(cv2.imdecode(np.asarray(x, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # ALOHA compressed
                elif bgr: x = x[..., ::-1]
                if a.flip: x = x[::-1, ::-1]
                yield x
        return it
    task = a.task or str(f.attrs.get("language_instruction", f.attrs.get("task", "")))
    if isinstance(task, bytes): task = task.decode()
    cams_arg = dict(c.split("=", 1) for c in a.camera)
    if "data" in f and any(k.startswith("demo") for k in f["data"]):        # robomimic
        out = []
        for d in sorted(f["data"], key=lambda k: int(k.split("_")[-1])):
            g = f["data"][d]; obs = g["obs"]
            t = a.task or g.attrs.get("language_instruction", f["data"].attrs.get("language_instruction", task))
            t = t.decode() if isinstance(t, bytes) else str(t)
            keys = a.state or sorted(k for k in obs if obs[k].ndim == 2)
            cams = cams_arg or {(k[:-6] if k.endswith("_image") else k): k for k in obs if obs[k].ndim == 4}
            state = np.concatenate([obs[k][()].reshape(len(obs[k]), -1) for k in keys], 1)
            names = [f"{k}_{i}" for k in keys for i in range(int(np.prod(obs[k].shape[1:])))]
            act = g[a.action[0] if a.action else "actions"][()]
            out.append(dict(state=state, action=act, cameras={c: frames(obs[p], a.bgr) for c, p in cams.items()}, task=t,
                            state_names=names, action_names=None, report={"file": f"{path}:{d}", "steps": len(state)}))
        return out
    sk = a.state or ["observations/qpos"]; ak = a.action or ["action"]
    if "observations/images" in f and not cams_arg: cams_arg = {k: f"observations/images/{k}" for k in f["observations/images"]}
    for k in sk + ak + list(cams_arg.values()):
        if k not in f: sys.exit(f"{path}: no dataset '{k}'. Run 'inspect' to see what's inside.")
    state = np.concatenate([f[k][()].reshape(len(f[k]), -1) for k in sk], 1)
    action = np.concatenate([f[k][()].reshape(len(f[k]), -1) for k in ak], 1)
    return [dict(state=state, action=action, cameras={c: frames(f[p], a.bgr) for c, p in cams_arg.items()}, task=task,
                 state_names=None, action_names=None, report={"file": str(path), "steps": len(state)})]

def _inputs(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir() and (p / "metadata.yaml").exists(): out.append(p)
        elif p.is_dir():
            subs = sorted(x for x in p.iterdir() if x.suffix in (".hdf5", ".h5", ".bag", ".mcap") or (x.is_dir() and (x / "metadata.yaml").exists()))
            out += subs
        else: out.append(p)
    return out

def cmd_to_lerobot(a):
    if not a.camera and not str(a.inputs[0]).endswith((".hdf5", ".h5")) and not Path(a.inputs[0]).is_dir():
        print("note: no --camera given, so the dataset will have no video")
    w = LeRobotWriter(a.out, a.fps, a.robot_type, codec=a.codec, crf=a.crf)
    reports = []
    for p in _inputs(a.inputs):
        print(p)
        eps = read_hdf5(p, a) if p.suffix in (".hdf5", ".h5") else read_log(p, a)
        for ep in eps:
            if not ep["cameras"]: sys.exit("at least one --camera is needed: the check reads video")
            w.add_episode(ep["state"], ep["action"], ep["cameras"], ep["task"], ep["state_names"], ep["action_names"])
            ep["report"]["episode"] = len(w.episodes) - 1; reports.append(ep["report"])
    meta = {"source": [str(x) for x in a.inputs]}
    for k in ("pose_dims", "angle_dims"):
        v = getattr(a, k)
        if v is not None: meta[k] = [int(x) for x in v.split(",") if x != ""]
    if a.gripper_dim is not None: meta.update(gripper_dim=a.gripper_dim, gripper_open=a.gripper_open, gripper_closed=a.gripper_closed)
    if a.spike_dims is not None: meta["spike_dims"] = [int(x) for x in a.spike_dims.split(",") if x != ""]
    w.close(meta)
    rep = pd.DataFrame(reports); cols = ["episode", "file", "steps"] + [c for c in rep.columns if c not in ("episode", "file", "steps")]
    rep[cols].to_csv(Path(a.out) / "conversion_report.csv", index=False)
    stale = [c for c in rep.columns if c.endswith("_stale_steps")]
    if stale and rep[stale].to_numpy().sum():
        print(f"warning: some steps had no message within one frame period; see {Path(a.out) / 'conversion_report.csv'}")

# ------------------------------------------------------------------ LeRobot -> MCAP for Foxglove

def _ranges(s, n):
    if not s: return list(range(n))
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-"); out += list(range(int(lo), int(hi or lo) + 1))
    return [e for e in out if e < n]

def cmd_to_mcap(a):
    import foxglove
    from foxglove.channels import CompressedImageChannel, LogChannel
    from foxglove.messages import CompressedImage, Log, LogLevel, Timestamp
    ds, out = Path(a.dataset), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = json.loads((ds / "meta/info.json").read_text()); fps = info["fps"]
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    eps = pd.read_parquet(ds / "meta/episodes/chunk-000/file-000.parquet").sort_values("episode_index").reset_index(drop=True)
    data = pd.read_parquet(ds / "data/chunk-000/file-000.parquet")
    names = lambda k: (info["features"][k].get("names") or {}).get("motors") if isinstance(info["features"][k].get("names"), dict) else info["features"][k].get("names")
    flags = pd.read_csv(a.flags) if a.flags else pd.DataFrame(columns=["episode", "code", "start_s", "end_s", "detail"])
    pick = _ranges(a.episodes, len(eps))
    base = 1_000_000_000  # start each episode 1 s after the epoch so no message has time zero
    def ts(s):
        t = base + int(round(s * 1e9)); return Timestamp(sec=t // 10**9, nsec=t % 10**9)
    vec_schema = {"type": "object", "properties": {"position": {"type": "array", "items": {"type": "number"}},
                                                   "names": {"type": "array", "items": {"type": "string"}}}}
    # decode every camera once, routing frames to the episodes we want
    frames = {e: {c: [] for c in cams} for e in pick}
    for cam in cams:
        windows = [(e, float(eps.loc[e, f"videos/{cam}/from_timestamp"]), float(eps.loc[e, f"videos/{cam}/to_timestamp"])) for e in pick]
        lo, hi = min(w[1] for w in windows), max(w[2] for w in windows)
        vp = next((ds / "videos" / cam).rglob("*.mp4"))
        c = av.open(str(vp)); st = c.streams.video[0]
        c.seek(int(max(0, lo - 1) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base)
            if t >= hi - 0.5 / fps: break
            for e, f0, f1 in windows:
                if f0 - 0.5 / fps <= t < f1 - 0.5 / fps:
                    ok, buf = cv2.imencode(".jpg", _frame_to_bgr(fr), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
                    frames[e][cam].append((t - f0, buf.tobytes()))
        c.close()
    for e in pick:
        er = eps.loc[e]; rows = data[data.episode_index == er.episode_index].sort_values("frame_index")
        path = out / f"episode_{int(er.episode_index):03d}.mcap"
        if path.exists(): path.unlink()
        ctx = foxglove.Context(); w = foxglove.open_mcap(str(path), context=ctx)
        task = (er["tasks"][0] if len(er["tasks"]) else "")
        w.write_metadata("episode", {"episode_index": str(int(er.episode_index)), "task": task, "fps": str(fps), "source": str(ds)})
        img_ch = {c: CompressedImageChannel(f"/camera/{c.split('.')[-1]}", context=ctx) for c in cams}
        sch = foxglove.Schema(name="fastagger.Vector", encoding="jsonschema", data=json.dumps(vec_schema).encode())
        st_ch = foxglove.Channel("/robot/state", schema=sch, message_encoding="json", context=ctx)
        ac_ch = foxglove.Channel("/robot/action", schema=sch, message_encoding="json", context=ctx)
        task_ch = LogChannel("/task", context=ctx); flag_ch = LogChannel("/fastagger/flags", context=ctx)
        lt = lambda s: base + int(round(s * 1e9))
        task_ch.log(Log(timestamp=ts(0), level=LogLevel.Info, message=task, name="task"), log_time=lt(0))
        sn, an = names("observation.state"), names("action")
        for _, r in rows.iterrows():
            t = float(r.frame_index) / fps
            st_ch.log(json.dumps({"position": [float(x) for x in r["observation.state"]], **({"names": sn} if sn else {})}).encode(), log_time=lt(t))
            ac_ch.log(json.dumps({"position": [float(x) for x in r["action"]], **({"names": an} if an else {})}).encode(), log_time=lt(t))
        for c in cams:
            for t, jpg in frames[e][c]:
                img_ch[c].log(CompressedImage(timestamp=ts(t), frame_id=c.split(".")[-1], data=jpg, format="jpeg"), log_time=lt(t))
        mine = flags[flags.episode == er.episode_index]
        for _, fr in mine.iterrows():
            flag_ch.log(Log(timestamp=ts(float(fr.start_s)), level=LogLevel.Warning, message=f"{fr.code}: {fr.detail} ({fr.start_s}s to {fr.end_s}s)",
                            name="fastagger"), log_time=lt(float(fr.start_s)))
        w.close()
        print(f"{path}: {len(rows)} steps, {sum(len(frames[e][c]) for c in cams)} images, {len(mine)} flags")

# ------------------------------------------------------------------ inspect

def cmd_inspect(a):
    p = Path(a.path)
    if p.suffix in (".hdf5", ".h5"):
        import h5py
        with h5py.File(p, "r") as f:
            for k, v in f.attrs.items(): print(f"@{k} = {v!r}"[:120])
            f.visititems(lambda n, o: print(f"{n}  {o.shape} {o.dtype}") if hasattr(o, "shape") else None)
        return
    stats = {}
    for topic, mtype, t, _ in _iter_messages(p):
        s = stats.setdefault(topic, [mtype, 0, t, t]); s[1] += 1; s[3] = t
    print(f"{'topic':40} {'type':40} {'messages':>9} {'rate Hz':>8}")
    for topic, (mtype, n, t0, t1) in sorted(stats.items()):
        print(f"{topic:40} {mtype:40} {n:9d} {((n - 1) / max((t1 - t0) / 1e9, 1e-9)):8.1f}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect", help="list the topics or datasets in a file"); i.add_argument("path")
    t = sub.add_parser("to-lerobot", help="convert HDF5, ROS bags or MCAP to a LeRobot v3 dataset")
    t.add_argument("inputs", nargs="+"); t.add_argument("--out", required=True); t.add_argument("--fps", type=int, required=True)
    t.add_argument("--state", action="append", default=[]); t.add_argument("--action", action="append")
    t.add_argument("--camera", action="append", default=[], help="name=topic or name=dataset path")
    t.add_argument("--task", default=""); t.add_argument("--task-topic")
    t.add_argument("--stamp", choices=["log", "header"], default="log", help="time messages by when they were logged (default) or by header.stamp")
    t.add_argument("--robot-type", default="unknown"); t.add_argument("--codec", choices=["libx264", "libsvtav1"], default="libx264")
    t.add_argument("--crf", type=int, default=18, help="video quality, lower is better (default 18)")
    t.add_argument("--bgr", action="store_true", help="HDF5 images are stored BGR"); t.add_argument("--flip", action="store_true", help="HDF5 images are upside down (robosuite)")
    g = t.add_argument_group("hints for the check (saved in meta/fastagger.json)")
    g.add_argument("--pose-dims", help="state dims that describe the arm's position, e.g. 0,1,2,3,4,5 (default: all but the gripper)")
    g.add_argument("--angle-dims", help="state dims that are angles which wrap at +/-pi")
    g.add_argument("--gripper-dim", type=int, help="state dim of the gripper; leave out if there is none")
    g.add_argument("--gripper-open", type=float, default=0.0); g.add_argument("--gripper-closed", type=float, default=1.0)
    g.add_argument("--spike-dims", help="action dims to check for one-step spikes (default: the pose dims)")
    m = sub.add_parser("to-mcap", help="export LeRobot v3 episodes as MCAP files for Foxglove")
    m.add_argument("dataset"); m.add_argument("--out", required=True); m.add_argument("--episodes", help="e.g. 0-19 or 3,12,40")
    m.add_argument("--flags", help="events.csv from fastagger_check.py analyze, shown on /fastagger/flags")
    m.add_argument("--jpeg-quality", type=int, default=90)
    a = ap.parse_args()
    {"inspect": cmd_inspect, "to-lerobot": cmd_to_lerobot, "to-mcap": cmd_to_mcap}[a.cmd](a)

if __name__ == "__main__":
    main()
