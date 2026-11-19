# Fastagger Check

A first-pass quality check for robot learning datasets in the LeRobot v3 format. It reads a dataset, looks at the robot's state and actions and at every camera's video, and flags episodes worth a second look before you train on them: missing instructions, cut-short recordings, long idle stretches, one-step action spikes, possible dropped objects, frozen or missing video, and cameras whose video lags the arm.

It only reads the dataset. It never changes your episodes.

It was built for, and tested on, [DROID's public 100-episode sample](https://huggingface.co/datasets/lerobot/droid_100). The results of that run are in [`examples/droid_100`](examples/droid_100).

## Quick start

```bash
pip install -r requirements.txt

# 1. Decode each camera once and cache per-frame motion and brightness
python fastagger_check.py decode  path/to/droid_100 out observation.images.exterior_image_1_left
python fastagger_check.py decode  path/to/droid_100 out observation.images.exterior_image_2_left
python fastagger_check.py decode  path/to/droid_100 out observation.images.wrist_image_left

# 2. Run the checks
python fastagger_check.py analyze path/to/droid_100 out

# 3. Optional: one verdict per episode, and example frames for flagged events
python verdicts.py out
python fastagger_check.py strips  path/to/droid_100 out
```

To get the sample, download `lerobot/droid_100` from Hugging Face, for example with `huggingface-cli download lerobot/droid_100 --repo-type dataset --local-dir droid_100`.

## What it writes

| File | Contents |
|---|---|
| `per_episode.csv` | One row per episode: task text, length, idle time, action spikes, grasps, timestamp gaps, missing values, duplicates, and for each camera its frame count, frozen runs, dark frames and video lag (in frames, with correlation r) |
| `events.csv` | Every flag raised: episode, start and end in seconds, and a one-line detail |
| `summary.json` | Totals for the run: flags by type, episode lengths and timing |
| `episode_verdicts.csv` | From `verdicts.py`: one verdict per episode |
| `frames/` | From `strips`: all cameras around a flagged moment, labeled with the flag |

## The checks

All thresholds are in `fastagger_check.py`. They were tuned on DROID, so expect to adjust them for other robots.

| Flag | What raises it |
|---|---|
| `MISSING_INSTRUCTION` | The episode has no language instruction |
| `TRUNCATED` | The episode is shorter than 2 seconds |
| `TIMESTAMP_GAP` | A step longer than 1.5 frame periods, or a timestamp that doesn't increase |
| `MISSING_DATA` | Missing (NaN) values in the state or actions |
| `LONG_IDLE` | The arm doesn't move for 3 seconds or more. Motion is the change in arm pose per step, with angles unwrapped, scaled by the dataset-wide spread |
| `ACTION_SPIKE` | One of the first six action values jumps by more than 0.5 for a single step and jumps straight back |
| `POSSIBLE_DROP` | The gripper holds steady, part-closed, then snaps fully shut within about half a second. This can mean a lost object |
| `MANY_GRASPS` | The gripper closes three or more times in one episode |
| `DUPLICATE` | The episode's state trajectory is identical to an earlier episode's |
| `DROPPED_FRAMES` | A camera has more or fewer video frames than the episode has steps |
| `FROZEN_VIDEO` | A camera's picture doesn't change for 8 or more frames while the arm is moving |
| `SYNC_OFFSET` | The wrist camera's motion best matches the arm's motion at a lag of 2 or more frames, with correlation r of at least 0.4 |

Dark frames are counted per camera but not flagged.

Flags are candidates, not confirmed faults. On DROID, the drop flags we reviewed frame by frame were objects placed on purpose.

## Verdicts

`verdicts.py` gives each episode its most serious bucket:

| Verdict | Flags |
|---|---|
| Re-record | `TRUNCATED` |
| For a person | `POSSIBLE_DROP`, `MANY_GRASPS`, `SYNC_OFFSET`, `FROZEN_VIDEO`, `DROPPED_FRAMES`, `DUPLICATE` |
| Software fix | `MISSING_INSTRUCTION`, `LONG_IDLE`, `ACTION_SPIKE`, `TIMESTAMP_GAP`, `MISSING_DATA` |
| Clean | No flags |

`FROZEN_VIDEO`, `DROPPED_FRAMES`, `DUPLICATE`, `TIMESTAMP_GAP` and `MISSING_DATA` didn't occur on DROID, so their buckets haven't been tested on real data.

## Results on DROID's 100-episode sample

On a laptop, the whole run took 51 seconds: 50.6 s to decode the video and 0.4 s for the checks, about 0.5 s per episode.

- **Verdicts:** 33 Clean, 46 Software fix, 20 For a person, 1 Re-record.
- **Flags:** MISSING_INSTRUCTION 54, MANY_GRASPS 18, POSSIBLE_DROP 4, LONG_IDLE 3, ACTION_SPIKE 2, SYNC_OFFSET 1, TRUNCATED 1.
- **Camera timing:** measurable in 96 episodes, where the wrist video and arm motion correlate at r ≥ 0.4. 79 show no lag. Episode 12's wrist video lags the arm by 3 frames (200 ms) at r = 0.82, not yet confirmed by a person.
- **Task success:** no person has yet judged whether each task succeeded.

## Requirements

Python 3.10 or later. Tested with numpy 2.4.6, pandas 3.0.6, pyarrow 26.0.0, av 18.1.0 and OpenCV 5.0.0. `av` must be able to decode the dataset's video codec; DROID's videos are AV1.

## Data credit

The example results and frames in `examples/droid_100` come from DROID (Khazatsky et al., 2024), through the `lerobot/droid_100` sample on Hugging Face, under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. See <https://droid-dataset.github.io/>.

## Website

`docs/index.html` is a one-page site for GitHub Pages. Set your repository link and Google Form link in the three lines marked `EDIT THESE` at the top of the file.

## License

The code is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Fastagger Inc.

Questions: Mutembei Kariuki, Fastagger, mkariuki@fastagger.com
