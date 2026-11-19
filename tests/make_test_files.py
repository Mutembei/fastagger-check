# Copyright 2026 Fastagger Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repackage episodes of a LeRobot v3 dataset into the other formats fastagger_convert.py reads, so every
converter can be tested on real robot data with a known answer.

python tests/make_test_files.py <lerobot dataset> <out folder> [episodes, default 0-9]

Writes: aloha/ (HDF5, raw images), aloha_jpeg/ (HDF5, JPEG-compressed images), robomimic.hdf5,
ros1/*.bag, ros2/<episode>/ (sqlite3), ros2_mcap/<episode>/ (MCAP storage). Bag timestamps get up to
10 ms of random jitter, as on a real robot.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, av, cv2, h5py

ds, out = Path(sys.argv[1]), Path(sys.argv[2])
lo, _, hi = (sys.argv[3] if len(sys.argv) > 3 else "0-9").partition("-"); pick = list(range(int(lo), int(hi or lo) + 1))
info = json.loads((ds / "meta/info.json").read_text()); fps = info["fps"]
cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
eps = pd.read_parquet(ds / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
data = pd.read_parquet(ds / "data/chunk-000/file-000.parquet")
rng = np.random.default_rng(0)

def bgr(fr):
    if fr.format.name == "yuv420p" and fr.color_range != 2: return cv2.cvtColor(fr.to_ndarray(format="yuv420p"), cv2.COLOR_YUV2BGR_I420)
    return fr.to_ndarray(format="bgr24")

def frames(cam, f0, n):
    c = av.open(str(next((ds / "videos" / cam).rglob("*.mp4")))); s = c.streams.video[0]; got = []
    c.seek(int(max(0, f0 - 1) / s.time_base), stream=s, backward=True)
    for fr in c.decode(s):
        if float(fr.pts * s.time_base) < f0 - 0.5 / fps: continue
        got.append(bgr(fr)[..., ::-1]);
        if len(got) == n: break
    c.close(); return np.stack(got)

episodes = []
for e in pick:
    rows = data[data.episode_index == e].sort_values("frame_index")
    st, ac = np.stack(rows["observation.state"]).astype(np.float64), np.stack(rows["action"]).astype(np.float64)
    imgs = {c.split(".")[-1]: frames(c, float(eps.loc[e, f"videos/{c}/from_timestamp"]), len(rows)) for c in cams}
    task = eps.loc[e, "tasks"][0] if len(eps.loc[e, "tasks"]) else ""
    episodes.append((e, st, ac, imgs, task)); print("read episode", e, len(rows), "steps")

for d in ("aloha", "aloha_jpeg", "ros1", "ros2", "ros2_mcap"): (out / d).mkdir(parents=True, exist_ok=True)
for e, st, ac, imgs, task in episodes:
    with h5py.File(out / f"aloha/episode_{e}.hdf5", "w") as f:
        f.attrs["sim"] = False; f.attrs["language_instruction"] = task
        f["observations/qpos"] = st; f["action"] = ac
        for c, im in imgs.items(): f[f"observations/images/{c}"] = im
    with h5py.File(out / f"aloha_jpeg/episode_{e}.hdf5", "w") as f:
        f.attrs["compress"] = True; f.attrs["language_instruction"] = task
        f["observations/qpos"] = st; f["action"] = ac
        for c, im in imgs.items():
            enc = [cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1] for x in im]
            pad = np.zeros((len(enc), max(len(x) for x in enc)), np.uint8)
            for i, x in enumerate(enc): pad[i, :len(x)] = x
            f[f"observations/images/{c}"] = pad
with h5py.File(out / "robomimic.hdf5", "w") as f:
    for i, (e, st, ac, imgs, task) in enumerate(episodes):
        g = f.create_group(f"data/demo_{i}"); g["obs/robot_state"] = st; g["actions"] = ac; g.attrs["language_instruction"] = task
        for c, im in imgs.items(): g[f"obs/{c}_image"] = im

from rosbags.typesys import Stores, get_typestore
from rosbags.rosbag1 import Writer as W1
from rosbags.rosbag2 import Writer as W2, StoragePlugin

def write_bag(kind, path, e, st, ac, imgs, task):
    store = get_typestore(Stores.ROS1_NOETIC if kind == "ros1" else Stores.ROS2_HUMBLE); T = store.types
    Time, Header = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"]
    hdr = lambda ns, fid="": Header(seq=0, stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9), frame_id=fid) if kind == "ros1" else Header(stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9), frame_id=fid)
    ser = store.serialize_ros1 if kind == "ros1" else store.serialize_cdr
    w = W1(path) if kind == "ros1" else W2(path, version=8, storage_plugin=StoragePlugin.MCAP if kind == "ros2_mcap" else StoragePlugin.SQLITE3)
    w.open()
    con = lambda topic, t: w.add_connection(topic, t, typestore=store)
    cj, ca, ci = con("/joint_states", "sensor_msgs/msg/JointState"), con("/arm/command", "std_msgs/msg/Float64MultiArray"), con("/instruction", "std_msgs/msg/String")
    names = list(imgs)
    cc = {names[0]: con(f"/{names[0]}/image_raw", "sensor_msgs/msg/Image"),          # raw RGB
          names[1]: con(f"/{names[1]}/image_raw", "sensor_msgs/msg/Image"),          # raw BGR
          names[2]: con(f"/{names[2]}/image_raw/compressed", "sensor_msgs/msg/CompressedImage")}
    t0 = 1_700_000_000 * 10**9; jit = lambda: int(rng.uniform(-10e6, 10e6))
    msgs = [(t0 - 10**6, ci, T["std_msgs/msg/String"](data=task))]
    Layout, Dim = T["std_msgs/msg/MultiArrayLayout"], T["std_msgs/msg/MultiArrayDimension"]
    for k in range(len(st)):
        tk = t0 + int(round(k * 1e9 / fps))
        msgs.append((tk + jit(), cj, T["sensor_msgs/msg/JointState"](header=hdr(tk), name=[f"joint_{j}" for j in range(st.shape[1])],
                     position=st[k], velocity=np.zeros(0), effort=np.zeros(0))))
        msgs.append((tk + jit(), ca, T["std_msgs/msg/Float64MultiArray"](layout=Layout(dim=[], data_offset=0), data=ac[k])))
        for c, im in imgs.items():
            x = im[k]
            if "compressed" in cc[c].topic:
                m = T["sensor_msgs/msg/CompressedImage"](header=hdr(tk, c), format="jpeg", data=cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].ravel())
            else:
                rgb = c == names[0]; px = x if rgb else x[..., ::-1]
                m = T["sensor_msgs/msg/Image"](header=hdr(tk, c), height=x.shape[0], width=x.shape[1], encoding="rgb8" if rgb else "bgr8",
                                               is_bigendian=0, step=x.shape[1] * 3, data=np.ascontiguousarray(px).ravel())
            msgs.append((tk + jit(), cc[c], m))
    for t, c, m in sorted(msgs, key=lambda x: x[0]): w.write(c, t, ser(m, c.msgtype))
    w.close()

for e, st, ac, imgs, task in episodes:
    write_bag("ros1", out / f"ros1/episode_{e}.bag", e, st, ac, imgs, task)
    write_bag("ros2", out / f"ros2/episode_{e}", e, st, ac, imgs, task)
    write_bag("ros2_mcap", out / f"ros2_mcap/episode_{e}", e, st, ac, imgs, task)
    print("wrote bags for episode", e)
