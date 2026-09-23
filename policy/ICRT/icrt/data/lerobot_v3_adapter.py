"""Read the local Piper LeRobot v3 dataset through ICRT's dataset interface.

The recorded Cartesian fields are deliberately not used here.  They were
computed with different kinematic definitions on the leader and follower.
Both proprioception and action targets are reconstructed from joint angles
with the same URDF and the same end-effector link.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

from .utils import rot_mat_to_rot_6d


def _resolve_path(value: str, config_path: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path.resolve()


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw rotation: Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


class UrdfForwardKinematics:
    """Small URDF FK implementation for a serial revolute chain."""

    def __init__(self, urdf_path: str | Path, base_link: str, end_link: str):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        child_to_joint = {}
        for joint in root.findall("joint"):
            child_to_joint[joint.find("child").attrib["link"]] = joint

        chain = []
        link = end_link
        while link != base_link:
            if link not in child_to_joint:
                raise ValueError(f"No URDF chain from {base_link!r} to {end_link!r}; stopped at {link!r}")
            joint = child_to_joint[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        self.chain = list(reversed(chain))
        self.moving_joints = [j for j in self.chain if j.attrib.get("type", "fixed") != "fixed"]

    @staticmethod
    def _floats(value: str | None, default: str) -> np.ndarray:
        return np.fromstring(value if value is not None else default, sep=" ", dtype=np.float64)

    def forward(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (len(self.moving_joints),):
            raise ValueError(f"Expected {len(self.moving_joints)} joints, got {joints.shape}")

        transform = np.eye(4, dtype=np.float64)
        moving_index = 0
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = self._floats(origin.attrib.get("xyz") if origin is not None else None, "0 0 0")
            rpy = self._floats(origin.attrib.get("rpy") if origin is not None else None, "0 0 0")
            fixed = np.eye(4, dtype=np.float64)
            fixed[:3, :3] = _rpy_matrix(rpy)
            fixed[:3, 3] = xyz
            transform = transform @ fixed

            joint_type = joint.attrib.get("type", "fixed")
            if joint_type == "fixed":
                continue
            axis_element = joint.find("axis")
            axis = self._floats(
                axis_element.attrib.get("xyz") if axis_element is not None else None,
                "1 0 0",
            )
            motion = np.eye(4, dtype=np.float64)
            if joint_type in {"revolute", "continuous"}:
                motion[:3, :3] = _axis_angle_matrix(axis, float(joints[moving_index]))
            elif joint_type == "prismatic":
                motion[:3, 3] = axis * float(joints[moving_index])
            else:
                raise ValueError(f"Unsupported URDF joint type: {joint_type}")
            transform = transform @ motion
            moving_index += 1
        return transform[:3, 3], transform[:3, :3]

    def forward_batch(self, joints: np.ndarray) -> np.ndarray:
        poses = []
        for joint_row in np.asarray(joints):
            translation, rotation = self.forward(joint_row)
            poses.append(np.concatenate([translation, rot_mat_to_rot_6d(rotation[None])[0]]))
        return np.asarray(poses, dtype=np.float32)


def trim_trailing_duplicate_action_timestamps(rows: np.ndarray, action_timestamps: np.ndarray) -> np.ndarray:
    """Remove the fixed-duration tail after the final fresh leader command.

    Short duplicate runs inside a demonstration are retained because they can
    represent a deliberate hold.  The first row of the final held command is
    retained as the terminal state/action pair.
    """
    if len(rows) < 2:
        return rows
    timestamps = action_timestamps[rows]
    final_timestamp = timestamps[-1]
    tail_start = len(timestamps) - 1
    while tail_start > 0 and timestamps[tail_start - 1] == final_timestamp:
        tail_start -= 1
    return rows[: tail_start + 1]


class LeRobotV3PiperAdapter:
    """Expose a LeRobot v3 Piper recording as ICRT episode/key arrays."""

    CARTESIAN_KEYS = {
        "observation/cartesian_position": "proprio_pose",
        "action/cartesian_position": "action_pose",
    }
    GRIPPER_KEYS = {
        "observation/gripper_position": "proprio_gripper",
        "action/gripper_position": "action_gripper",
    }

    def __init__(self, dataset_json: dict, dataset_config_path: str | Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("LeRobot v3 loading requires pyarrow. Reinstall ICRT after updating pyproject.toml.") from exc

        self.root = _resolve_path(dataset_json["dataset_path"], dataset_config_path)
        self.urdf_path = _resolve_path(dataset_json["urdf_path"], dataset_config_path)
        self.base_link = dataset_json.get("base_link", "base_link")
        self.end_link = dataset_json.get("end_link", "link6")
        self.target_fps = int(dataset_json.get("target_fps", 15))
        self.video_decode_short_side = int(dataset_json.get("video_decode_short_side", 256))
        self.image_keys = dataset_json["image_keys"]

        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.source_fps = int(info["fps"])
        if self.source_fps % self.target_fps != 0:
            raise ValueError(f"source fps {self.source_fps} must be divisible by target fps {self.target_fps}")
        self.frame_stride = self.source_fps // self.target_fps

        columns = [
            "index",
            "episode_index",
            "task_index",
            "timestamp.action_ns",
            "observation.state",
            "action.joint_absolute",
        ]
        tables = [pq.read_table(path, columns=columns) for path in sorted((self.root / "data").glob("chunk-*/*.parquet"))]
        if not tables:
            raise FileNotFoundError(f"No LeRobot parquet files found under {self.root / 'data'}")
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

        self.global_indices = self._column_to_numpy(table, "index").reshape(-1).astype(np.int64)
        episode_column = self._column_to_numpy(table, "episode_index").reshape(-1).astype(np.int64)
        task_column = self._column_to_numpy(table, "task_index").reshape(-1).astype(np.int64)
        action_timestamps = self._column_to_numpy(table, "timestamp.action_ns").reshape(-1).astype(np.int64)
        follower_state = self._column_to_numpy(table, "observation.state").astype(np.float64)
        leader_target = self._column_to_numpy(table, "action.joint_absolute").astype(np.float64)

        if follower_state.shape[1] != 7 or leader_target.shape[1] != 7:
            raise ValueError("Expected 6 Piper joints plus one gripper value for state and action")

        fk = UrdfForwardKinematics(self.urdf_path, self.base_link, self.end_link)
        self.proprio_pose = fk.forward_batch(follower_state[:, :6])
        self.action_pose = fk.forward_batch(leader_target[:, :6])
        self.proprio_gripper = follower_state[:, 6:7].astype(np.float32)
        self.action_gripper = leader_target[:, 6:7].astype(np.float32)

        self.episode_rows: Dict[str, np.ndarray] = {}
        task_names = self._load_task_names(pq, dataset_json)
        self.verb_to_episode = {name: [] for name in task_names.values()}
        trimmed = 0
        for episode_index in sorted(np.unique(episode_column)):
            rows = np.flatnonzero(episode_column == episode_index)
            episode_tasks = np.unique(task_column[rows])
            if len(episode_tasks) != 1:
                raise ValueError(
                    f"Episode {episode_index} contains multiple task labels {episode_tasks.tolist()}; "
                    "ICRT expects one task label per demonstration."
                )

            cleaned_rows = rows
            # DATA CLEANING: comment out this one line for already-clean future recordings.
            # cleaned_rows = trim_trailing_duplicate_action_timestamps(rows, action_timestamps)

            trimmed += len(rows) - len(cleaned_rows)
            sampled_rows = cleaned_rows[:: self.frame_stride]
            episode_id = f"episode_{int(episode_index):06d}"
            self.episode_rows[episode_id] = sampled_rows
            task_index = int(episode_tasks[0])
            if task_index not in task_names:
                raise KeyError(f"task_index {task_index} is missing from meta/tasks.parquet")
            self.verb_to_episode.setdefault(task_names[task_index], []).append(episode_id)

        self.episode_ids = sorted(self.episode_rows)
        self.episode_lengths = {key: len(rows) for key, rows in self.episode_rows.items()}
        self.verb_to_episode = {task: episodes for task, episodes in self.verb_to_episode.items() if episodes}
        self.video_paths = self._find_video_paths()
        self._video_containers = {}

        print(
            f"Loaded LeRobot v3 dataset: {len(self.episode_ids)} episodes, "
            f"trimmed {trimmed} trailing stale frames, downsampled "
            f"{self.source_fps} Hz -> {self.target_fps} Hz"
        )

    @staticmethod
    def _column_to_numpy(table, name: str) -> np.ndarray:
        column = table[name].combine_chunks()
        if hasattr(column.type, "list_size"):
            return np.asarray(column.values).reshape(len(column), column.type.list_size)
        return np.asarray(column)

    def _load_task_names(self, parquet_module, dataset_json: dict) -> Dict[int, str]:
        tasks_path = self.root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            fallback = dataset_json.get("task_name")
            if fallback is None:
                raise FileNotFoundError(f"Missing {tasks_path} and no fallback task_name was configured")
            return {0: fallback}

        task_table = parquet_module.read_table(tasks_path)
        if "task_index" not in task_table.column_names:
            raise ValueError(f"{tasks_path} does not contain task_index")
        label_columns = [name for name in task_table.column_names if name != "task_index"]
        if not label_columns:
            raise ValueError(f"{tasks_path} does not contain a task label column")
        # LeRobot v3 stores the task string in the parquet index column.
        label_column = "__index_level_0__" if "__index_level_0__" in label_columns else label_columns[0]
        indices = task_table["task_index"].to_pylist()
        labels = task_table[label_column].to_pylist()
        return {int(index): str(label) for index, label in zip(indices, labels)}

    def _find_video_paths(self) -> Dict[str, Path]:
        paths = {}
        for key in self.image_keys:
            matches = sorted((self.root / "videos" / key).glob("chunk-*/*.mp4"))
            if len(matches) != 1:
                raise ValueError(
                    f"This adapter currently expects one MP4 for {key!r}; found {len(matches)} under {self.root}"
                )
            paths[key] = matches[0]
        return paths

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_containers"] = {}
        return state

    def _video_container(self, key: str):
        try:
            import av
        except ImportError as exc:
            raise ImportError("LeRobot AV1 video loading requires PyAV (`pip install av`).") from exc
        container = self._video_containers.get(key)
        if container is None:
            container = av.open(str(self.video_paths[key]))
            self._video_containers[key] = container
        return container

    @staticmethod
    def _contiguous_runs(indices: Iterable[int], maximum_gap: int) -> List[List[int]]:
        sorted_indices = sorted(set(int(i) for i in indices))
        if not sorted_indices:
            return []
        runs = [[sorted_indices[0]]]
        for index in sorted_indices[1:]:
            if index - runs[-1][-1] <= maximum_gap:
                runs[-1].append(index)
            else:
                runs.append([index])
        return runs

    def _read_video(self, key: str, global_indices: np.ndarray) -> np.ndarray:
        container = self._video_container(key)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        wanted = [int(index) for index in global_indices]
        positions = {}
        for position, frame_index in enumerate(wanted):
            positions.setdefault(frame_index, []).append(position)
        decoded_indices = set()
        output = None
        for run in self._contiguous_runs(wanted, self.frame_stride):
            wanted_in_run = set(run)
            seek_pts = int((run[0] / fps) / float(stream.time_base))
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
            for frame in container.decode(stream):
                frame_index = int(round(float(frame.pts * stream.time_base) * fps))
                if frame_index < run[0]:
                    continue
                if frame_index > run[-1]:
                    break
                if frame_index in wanted_in_run:
                    image = frame.to_ndarray(format="rgb24")
                    height, width = image.shape[:2]
                    scale = self.video_decode_short_side / min(height, width)
                    resized = cv2.resize(
                        image,
                        (round(width * scale), round(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                    if output is None:
                        output = np.empty((len(wanted), *resized.shape), dtype=np.uint8)
                    for position in positions[frame_index]:
                        output[position] = resized
                    decoded_indices.add(frame_index)
        missing = sorted(set(wanted) - decoded_indices)
        if missing:
            raise RuntimeError(f"Missing {len(missing)} frames from {self.video_paths[key]}: {missing[:5]}")
        if output is None:
            raise RuntimeError(f"No frames decoded from {self.video_paths[key]}")
        return output

    def get(self, episode_id: str, key: str, begin: int, end: int) -> np.ndarray:
        rows = self.episode_rows[episode_id][begin : end + 1]
        if key in self.CARTESIAN_KEYS:
            return getattr(self, self.CARTESIAN_KEYS[key])[rows]
        if key in self.GRIPPER_KEYS:
            return getattr(self, self.GRIPPER_KEYS[key])[rows]
        if key in self.video_paths:
            return self._read_video(key, self.global_indices[rows])
        raise KeyError(f"Unsupported LeRobot v3 key: {key}")
