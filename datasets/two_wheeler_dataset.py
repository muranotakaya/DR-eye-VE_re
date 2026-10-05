"""Datasets for rider gaze saliency prediction (myEye2Wheeler-style data).

Two datasets share the same preprocessing (``_ClipSaliencyDataset``):

* ``TwoWheelerGazeDataset`` -- reads videos (or extracted frames) and gaze annotations from
  disk.
* ``DummyTwoWheelerDataset`` -- procedurally generates driving-like clips and a gaze track
  that follows a moving "target" object, so the whole pipeline can be tested without data.

Each sample is a clip of ``clip_len`` consecutive frames ending at frame ``t``; the target
is the gaze saliency map of frame ``t`` (the last frame, as in DR(eye)VE).

Expected on-disk layout (ASSUMPTION -- the official myEye2Wheeler release format has not
been verified here; adapt ``gaze_columns`` / ``coords`` in the config, or convert the raw
Tobii export into this layout)::

    <root>/
      splits/{train,val,test}.txt     # one sequence id per line (optional, see config)
      <seq_id>/
        video.mp4                     # scene-camera video, or
        frames/000000.jpg ...         # pre-extracted frames (faster random access)
        gaze.csv                      # gaze samples, columns: frame,x,y (see below)
        saliency/000000.png ...       # optional pre-computed GT maps (override gaze.csv)

``gaze.csv`` may contain several rows per frame (eye tracker rate > video fps) and rows
with empty/NaN x,y (lost tracking). ``x, y`` are normalized image coordinates in [0, 1]
(origin top-left; Tobii "gaze2d" convention) when ``coords: normalized``, or pixels of the
original video when ``coords: pixel``. Instead of ``frame``, a ``timestamp`` column in
seconds can be given; it is converted with the video fps.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Kinetics-400 statistics used by torchvision video models.
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)

FRAME_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_NAMES = ("video.mp4", "video.avi", "video.mov", "video.mkv")


# --------------------------------------------------------------------------------------
# Ground-truth generation
# --------------------------------------------------------------------------------------
def gaze_to_saliency(points: np.ndarray, size: Tuple[int, int], sigma: float,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Render normalized gaze points as a sum of isotropic Gaussians.

    :param points: [N, 2] array of (x, y) in normalized [0, 1] coordinates.
    :param size: (H, W) of the output map.
    :param sigma: Gaussian std as a fraction of the output width.
    :param weights: optional [N] weights (e.g. temporal decay).
    :return: float32 map [H, W] scaled to max 1 (all zeros if there are no points).
    """
    h, w = size
    out = np.zeros((h, w), dtype=np.float32)
    if points is None or len(points) == 0:
        return out
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    s = max(sigma * w, 1e-3)
    xs = (np.arange(w) + 0.5)[None, :]
    ys = (np.arange(h) + 0.5)[None, :]
    gx = np.exp(-((xs - pts[:, :1] * w) ** 2) / (2 * s * s))   # [N, W]
    gy = np.exp(-((ys - pts[:, 1:] * h) ** 2) / (2 * s * s))   # [N, H]
    if weights is not None:
        gy = gy * np.asarray(weights, dtype=np.float64)[:, None]
    out = (gy.T @ gx).astype(np.float32)                       # separable sum of Gaussians
    m = out.max()
    return out / m if m > 0 else out


def gaze_to_fixation_map(points: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Binary map with ones at the (rounded) gaze locations, used for NSS."""
    h, w = size
    out = np.zeros((h, w), dtype=np.float32)
    if points is None or len(points) == 0:
        return out
    pts = np.asarray(points).reshape(-1, 2)
    xi = np.clip((pts[:, 0] * w).astype(int), 0, w - 1)
    yi = np.clip((pts[:, 1] * h).astype(int), 0, h - 1)
    out[yi, xi] = 1.0
    return out


# --------------------------------------------------------------------------------------
# Shared preprocessing
# --------------------------------------------------------------------------------------
class _ClipSaliencyDataset(Dataset):
    """Common sampling / preprocessing. Subclasses provide frames and gaze points."""

    def __init__(self, clip_len: int = 16, frame_size: Sequence[int] = (448, 448),
                 train: bool = False, use_crop: bool = True,
                 crop_before_size: Sequence[int] = (256, 256), crop_type: str = "random",
                 mirror: bool = True, sigma: float = 0.03, gaze_window: int = 0,
                 mean: Sequence[float] = KINETICS_MEAN, std: Sequence[float] = KINETICS_STD,
                 seed: int = 0):
        self.clip_len = int(clip_len)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.train = train
        self.use_crop = use_crop and train
        self.crop_before_size = (int(crop_before_size[0]), int(crop_before_size[1]))
        self.crop_size = (self.frame_size[0] // 4, self.frame_size[1] // 4)
        if self.use_crop and (self.crop_size[0] > self.crop_before_size[0]
                              or self.crop_size[1] > self.crop_before_size[1]):
            raise ValueError(f"crop size {self.crop_size} larger than crop_before_size "
                             f"{self.crop_before_size}")
        if crop_type not in ("random", "central"):
            raise ValueError(f"crop_type must be 'random' or 'central', got {crop_type}")
        self.crop_type = crop_type
        self.mirror = mirror and train
        self.sigma = sigma
        self.gaze_window = int(gaze_window)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1, 1)
        self.seed = seed
        self._epoch = 0
        # list of (sequence key, target frame index)
        self.samples: List[Tuple[str, int]] = []

    # --- to be implemented by subclasses -------------------------------------------------
    def load_frames(self, seq: str, indices: Sequence[int]) -> np.ndarray:
        """Return uint8 RGB frames [T, h, w, 3] (any native resolution)."""
        raise NotImplementedError

    def gaze_points(self, seq: str, frame: int, window: int = 0) -> np.ndarray:
        """Return normalized gaze points [N, 2] of frames ``frame - window .. frame + window``."""
        raise NotImplementedError

    def precomputed_saliency(self, seq: str, frame: int) -> Optional[np.ndarray]:
        """Optional pre-computed GT map [h, w] float in [0, 1]; ``None`` to render from gaze."""
        return None

    # --- common logic ------------------------------------------------------------------
    def set_epoch(self, epoch: int):
        """Make random augmentation differ across epochs while staying reproducible."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self._epoch, index))

    def _normalize(self, frames: np.ndarray) -> np.ndarray:
        x = frames.astype(np.float32).transpose(3, 0, 1, 2) / 255.0   # [C, T, h, w]
        return (x - self.mean) / self.std

    @staticmethod
    def _resize_frames(frames: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        if frames.shape[1:3] == (h, w):
            return frames
        return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames])

    def __getitem__(self, index: int) -> Dict:
        seq, t = self.samples[index]
        rng = self._rng(index)
        indices = list(range(t - self.clip_len + 1, t + 1))
        raw = self.load_frames(seq, indices)                          # [T, h, w, 3] uint8

        frames = self._resize_frames(raw, self.frame_size)
        gt = self.precomputed_saliency(seq, t)
        if gt is None:
            gt = gaze_to_saliency(self.gaze_points(seq, t, self.gaze_window), self.frame_size,
                                  self.sigma)
        else:
            gt = cv2.resize(gt.astype(np.float32), self.frame_size[::-1],
                            interpolation=cv2.INTER_LINEAR)
            m = gt.max()
            gt = gt / m if m > 0 else gt
        fix = gaze_to_fixation_map(self.gaze_points(seq, t), self.frame_size)

        sample = {
            "clip": self._normalize(frames),
            "saliency": gt[None],
            "fixation": fix[None],
        }

        if self.use_crop:
            hb, wb = self.crop_before_size
            hc, wc = self.crop_size
            if self.crop_type == "random":
                y0 = int(rng.integers(0, hb - hc + 1))
                x0 = int(rng.integers(0, wb - wc + 1))
            else:
                y0, x0 = (hb - hc) // 2, (wb - wc) // 2
            before = self._resize_frames(raw, (hb, wb))
            gt_before = cv2.resize(gt, (wb, hb), interpolation=cv2.INTER_LINEAR)
            sample["clip_crop"] = self._normalize(before[:, y0:y0 + hc, x0:x0 + wc])
            sample["saliency_crop"] = gt_before[None, y0:y0 + hc, x0:x0 + wc]

        if self.mirror and rng.random() < 0.5:
            sample = {k: np.flip(v, axis=-1) for k, v in sample.items()}

        out = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
               for k, v in sample.items()}
        out["seq"] = seq
        out["frame"] = t
        return out


# --------------------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------------------
def read_split(root: Path, split: str) -> List[str]:
    path = root / "splits" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path} (or set data.{split}_sequences)")
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


class TwoWheelerGazeDataset(_ClipSaliencyDataset):
    """Clips + gaze saliency GT from videos / frames and ``gaze.csv`` files.

    :param root: dataset root (see module docstring for the expected layout).
    :param sequences: sequence ids; if ``None`` they are read from ``splits/<split>.txt``.
    :param split: split name used when ``sequences`` is ``None``.
    :param stride: distance between consecutive target frames.
    :param skip_no_gaze: drop target frames without any valid gaze sample.
    :param coords: ``normalized`` or ``pixel`` gaze coordinates.
    :param gaze_columns: mapping of ``frame``/``timestamp``/``x``/``y`` to CSV column names.
    :param fps: video fps for ``timestamp`` -> frame conversion (default: read from video).
    """

    def __init__(self, root: str, sequences: Optional[Sequence[str]] = None, split: str = "train",
                 stride: int = 1, skip_no_gaze: bool = True, coords: str = "normalized",
                 gaze_columns: Optional[Dict[str, str]] = None, fps: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        if coords not in ("normalized", "pixel"):
            raise ValueError(f"coords must be 'normalized' or 'pixel', got {coords}")
        self.coords = coords
        self.columns = {"frame": "frame", "timestamp": "timestamp", "x": "x", "y": "y"}
        self.columns.update(gaze_columns or {})
        self.fps = fps
        seqs = list(sequences) if sequences else read_split(self.root, split)

        self._frame_files: Dict[str, List[Path]] = {}
        self._video: Dict[str, Path] = {}
        self._gaze: Dict[str, Dict[int, np.ndarray]] = {}
        self._sal_dir: Dict[str, Path] = {}
        self._caps: Dict[str, cv2.VideoCapture] = {}

        for seq in seqs:
            n_frames, size, seq_fps = self._index_sequence(seq)
            self._gaze[seq] = self._read_gaze(seq, size, seq_fps)
            for t in range(self.clip_len - 1, n_frames, max(1, int(stride))):
                if (skip_no_gaze and seq not in self._sal_dir
                        and len(self.gaze_points(seq, t, self.gaze_window)) == 0):
                    continue
                self.samples.append((seq, t))
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} for sequences {seqs}")

    # --- indexing -----------------------------------------------------------------------
    def _index_sequence(self, seq: str) -> Tuple[int, Tuple[int, int], Optional[float]]:
        seq_dir = self.root / seq
        frames_dir = seq_dir / "frames"
        if frames_dir.is_dir():
            files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in FRAME_EXTS)
            if not files:
                raise RuntimeError(f"No frames in {frames_dir}")
            self._frame_files[seq] = files
            first = cv2.imread(str(files[0]))
            n, size, fps = len(files), first.shape[:2], self.fps
        else:
            video = next((seq_dir / v for v in VIDEO_NAMES if (seq_dir / v).is_file()), None)
            if video is None:
                raise FileNotFoundError(f"Neither frames/ nor video.* found in {seq_dir}")
            self._video[seq] = video
            cap = cv2.VideoCapture(str(video))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            fps = self.fps or cap.get(cv2.CAP_PROP_FPS) or None
            cap.release()
        if (seq_dir / "saliency").is_dir():
            self._sal_dir[seq] = seq_dir / "saliency"
        return n, size, fps

    def _read_gaze(self, seq: str, size: Tuple[int, int], fps: Optional[float]) -> Dict[int, np.ndarray]:
        path = self.root / seq / "gaze.csv"
        if not path.is_file():
            if seq in self._sal_dir:
                return {}
            raise FileNotFoundError(f"Missing gaze annotations: {path}")
        c = self.columns
        per_frame: Dict[int, list] = {}
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    x, y = float(row[c["x"]]), float(row[c["y"]])
                except (KeyError, ValueError, TypeError):
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                if row.get(c["frame"]) not in (None, ""):
                    frame = int(float(row[c["frame"]]))
                elif row.get(c["timestamp"]) not in (None, ""):
                    if not fps:
                        raise ValueError(f"{path}: timestamp column needs fps (set data.fps)")
                    frame = int(round(float(row[c["timestamp"]]) * fps))
                else:
                    raise KeyError(f"{path}: needs a '{c['frame']}' or '{c['timestamp']}' column")
                if self.coords == "pixel":
                    x, y = x / size[1], y / size[0]
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    per_frame.setdefault(frame, []).append((x, y))
        return {k: np.asarray(v, dtype=np.float32) for k, v in per_frame.items()}

    # --- accessors ----------------------------------------------------------------------
    def gaze_points(self, seq: str, frame: int, window: int = 0) -> np.ndarray:
        g = self._gaze[seq]
        pts = [g[i] for i in range(frame - window, frame + window + 1) if i in g]
        return np.concatenate(pts) if pts else np.zeros((0, 2), dtype=np.float32)

    def precomputed_saliency(self, seq: str, frame: int) -> Optional[np.ndarray]:
        d = self._sal_dir.get(seq)
        if d is None:
            return None
        path = d / f"{frame:06d}.png"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img.astype(np.float32) / 255.0

    def load_frames(self, seq: str, indices: Sequence[int]) -> np.ndarray:
        if seq in self._frame_files:
            files = self._frame_files[seq]
            frames = [cv2.imread(str(files[i])) for i in indices]
        else:
            frames = self._read_video(seq, indices)
        return np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])

    def _read_video(self, seq: str, indices: Sequence[int]) -> List[np.ndarray]:
        # One capture per sequence and worker process (DataLoader workers fork the dataset).
        key = f"{os.getpid()}:{seq}"
        cap = self._caps.get(key)
        if cap is None:
            cap = self._caps[key] = cv2.VideoCapture(str(self._video[seq]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        frames = []
        for i in indices:
            ok, f = cap.read()
            if not ok:
                raise IOError(f"Could not read frame {i} of {self._video[seq]}")
            frames.append(f)
        return frames

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_caps"] = {}   # VideoCapture objects are not picklable
        return state


# --------------------------------------------------------------------------------------
# Synthetic data (--dummy)
# --------------------------------------------------------------------------------------
class SyntheticScene:
    """Procedural, deterministic driving-like video with a gaze track.

    A static road scene (sky, road, lane markings converging to a vanishing point) with a
    few moving "vehicles"; the simulated rider looks at vehicle 0 (red), with small noise
    and occasional glances at the vanishing point. Frames are rendered at ``native_size``.
    """

    def __init__(self, seed: int, n_frames: int, native_size: Tuple[int, int] = (144, 256),
                 n_objects: int = 3):
        self.rng = np.random.default_rng(seed)
        self.n_frames = n_frames
        self.h, self.w = native_size
        self.vp = np.array([0.5 + self.rng.uniform(-0.1, 0.1), 0.42])
        self.background = self._background()
        self.objects = []
        colors = [(220, 40, 40), (40, 90, 220), (240, 200, 40), (60, 200, 90)]
        for k in range(n_objects):
            self.objects.append(dict(
                color=colors[k % len(colors)],
                c0=self.rng.uniform([0.15, 0.5], [0.85, 0.85]),
                amp=self.rng.uniform([0.1, 0.03], [0.3, 0.1]),
                freq=self.rng.uniform(0.01, 0.04, size=2),
                phase=self.rng.uniform(0, 2 * np.pi, size=2),
                radius=self.rng.uniform(0.05, 0.09),
            ))
        noise = self.rng.normal(0, 0.01, size=(n_frames, 2))
        glance = self.rng.random(n_frames) < 0.05
        self._gaze = np.empty((n_frames, 2), dtype=np.float32)
        for t in range(n_frames):
            p = self.vp if glance[t] else self.object_center(0, t)
            self._gaze[t] = np.clip(p + noise[t], 0.0, 1.0)

    def _background(self) -> np.ndarray:
        h, w = self.h, self.w
        img = np.zeros((h, w, 3), dtype=np.uint8)
        hy = int(self.vp[1] * h)
        sky = np.linspace(200, 120, hy)[:, None, None] * np.array([0.6, 0.8, 1.0])  # [hy, 1, 3]
        img[:hy] = sky.astype(np.uint8)
        img[hy:] = (90, 90, 95)
        vp = (int(self.vp[0] * w), hy)
        for xb in (-0.4, 0.2, 0.8, 1.4):
            cv2.line(img, vp, (int(xb * w), h), (235, 235, 235), 1, cv2.LINE_AA)
        img = img.astype(np.int16) + self.rng.integers(-8, 9, size=img.shape, dtype=np.int16)
        return np.clip(img, 0, 255).astype(np.uint8)

    def object_center(self, k: int, t: int) -> np.ndarray:
        o = self.objects[k]
        return np.clip(o["c0"] + o["amp"] * np.sin(2 * np.pi * o["freq"] * t + o["phase"]), 0.05, 0.95)

    def frame(self, t: int) -> np.ndarray:
        img = self.background.copy()
        # draw far objects first (smaller y == farther)
        order = sorted(range(len(self.objects)), key=lambda k: self.object_center(k, t)[1])
        for k in order:
            o = self.objects[k]
            cx, cy = self.object_center(k, t)
            r = o["radius"] * (0.5 + cy)   # perspective: closer objects are bigger
            x0, y0 = int((cx - r) * self.w), int((cy - 0.6 * r) * self.h)
            x1, y1 = int((cx + r) * self.w), int((cy + 0.6 * r) * self.h)
            cv2.rectangle(img, (x0, y0), (x1, y1), o["color"], -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), 1)
        return img

    def gaze(self, t: int) -> np.ndarray:
        return self._gaze[t]


class DummyTwoWheelerDataset(_ClipSaliencyDataset):
    """In-memory synthetic dataset with the same interface as ``TwoWheelerGazeDataset``."""

    def __init__(self, num_sequences: int = 4, frames_per_sequence: int = 64, stride: int = 4,
                 native_size: Sequence[int] = (144, 256), scene_seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.scenes = {
            f"dummy_{i:03d}": SyntheticScene(scene_seed * 1000 + i, frames_per_sequence,
                                             tuple(native_size))
            for i in range(num_sequences)
        }
        for seq in self.scenes:
            for t in range(self.clip_len - 1, frames_per_sequence, max(1, int(stride))):
                self.samples.append((seq, t))

    def load_frames(self, seq: str, indices: Sequence[int]) -> np.ndarray:
        scene = self.scenes[seq]
        return np.stack([scene.frame(i) for i in indices])

    def gaze_points(self, seq: str, frame: int, window: int = 0) -> np.ndarray:
        scene = self.scenes[seq]
        lo, hi = max(0, frame - window), min(scene.n_frames - 1, frame + window)
        return np.stack([scene.gaze(i) for i in range(lo, hi + 1)])


def write_dummy_dataset(root: str, num_sequences: int = 3, frames_per_sequence: int = 48,
                        native_size: Sequence[int] = (144, 256), fps: float = 25.0,
                        use_frames_dir: bool = False, seed: int = 0) -> Path:
    """Write a synthetic dataset to disk in the layout expected by ``TwoWheelerGazeDataset``.

    Useful to exercise the real loader (video decoding, CSV parsing) end-to-end.
    Sequences are split round-robin into train / val / test.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    h, w = native_size
    for i in range(num_sequences):
        seq = f"seq_{i:03d}"
        seq_dir = root / seq
        seq_dir.mkdir(exist_ok=True)
        scene = SyntheticScene(seed * 1000 + i, frames_per_sequence, (h, w))
        if use_frames_dir:
            (seq_dir / "frames").mkdir(exist_ok=True)
            for t in range(frames_per_sequence):
                cv2.imwrite(str(seq_dir / "frames" / f"{t:06d}.png"),
                            cv2.cvtColor(scene.frame(t), cv2.COLOR_RGB2BGR))
        else:
            writer = cv2.VideoWriter(str(seq_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter could not open an mp4v stream; "
                                   "use use_frames_dir=True")
            for t in range(frames_per_sequence):
                writer.write(cv2.cvtColor(scene.frame(t), cv2.COLOR_RGB2BGR))
            writer.release()
        with (seq_dir / "gaze.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "x", "y"])
            for t in range(frames_per_sequence):
                if t % 17 == 5:          # simulate lost tracking
                    wr.writerow([t, "", ""])
                    continue
                x, y = scene.gaze(t)
                wr.writerow([t, f"{x:.5f}", f"{y:.5f}"])
        splits[("train", "val", "test")[i % 3] if num_sequences >= 3 else "train"].append(seq)
    (root / "splits").mkdir(exist_ok=True)
    for name, seqs in splits.items():
        (root / "splits" / f"{name}.txt").write_text("\n".join(seqs) + ("\n" if seqs else ""))
    return root


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
def build_dataset(cfg: dict, split: str, dummy: bool = False) -> _ClipSaliencyDataset:
    """Build a dataset from the ``data`` section of a config.

    :param cfg: ``data`` config dict.
    :param split: ``train`` / ``val`` / ``test``.
    :param dummy: use ``DummyTwoWheelerDataset`` instead of reading from disk.
    """
    train = split == "train"
    common = dict(
        clip_len=cfg.get("clip_len", 16),
        frame_size=cfg.get("frame_size", (448, 448)),
        train=train,
        use_crop=cfg.get("use_crop", True),
        crop_before_size=cfg.get("crop_before_size", (256, 256)),
        crop_type=cfg.get("crop_type", "random"),
        mirror=cfg.get("mirror", True),
        sigma=cfg.get("sigma", 0.03),
        gaze_window=cfg.get("gaze_window", 0),
        mean=cfg.get("mean", KINETICS_MEAN),
        std=cfg.get("std", KINETICS_STD),
        seed=cfg.get("seed", 0),
    )
    if dummy:
        d = cfg.get("dummy", {})
        n = {"train": d.get("train_sequences", 4), "val": d.get("val_sequences", 1),
             "test": d.get("test_sequences", 1)}[split]
        return DummyTwoWheelerDataset(
            num_sequences=n,
            frames_per_sequence=d.get("frames_per_sequence", 64),
            stride=d.get("stride", 4),
            native_size=d.get("native_size", (144, 256)),
            scene_seed={"train": 1, "val": 2, "test": 3}[split],
            **common,
        )
    return TwoWheelerGazeDataset(
        root=cfg["root"],
        sequences=cfg.get(f"{split}_sequences"),
        split=split,
        stride=cfg.get(f"{split}_stride", cfg.get("stride", 1)),
        skip_no_gaze=cfg.get("skip_no_gaze", True),
        coords=cfg.get("coords", "normalized"),
        gaze_columns=cfg.get("gaze_columns"),
        fps=cfg.get("fps"),
        **common,
    )
