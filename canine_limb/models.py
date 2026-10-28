"""Bundled RTMPose AP-10K and YOLOX-Tiny, using their original preprocessing."""

from pathlib import Path
import hashlib

import cv2
import numpy as np

from .runtime import ModelSession, TrackingError

MODEL_HASHES = {
    "rtmpose_ap10k.onnx": "1cfd1c86e0d9e5d5f95178bcd95ee9a4e8386a624cd3c57519f27ff58cac7f28",
    "yolox_tiny.onnx": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
}


def model_path(name, directory=None):
    root = Path(directory).expanduser() if directory else Path(__file__).parent / "models"
    path = root / name
    if not path.is_file():
        raise TrackingError(f"Model not found: {path}. Extract the complete project including models.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_HASHES[name]:
        raise TrackingError(f"Checksum mismatch for {path.name}; restore the original bundled model.")
    return path


def preprocess_pose(frame, box):
    x1, y1, x2, y2 = box
    if not np.isfinite(box).all() or x2 <= x1 or y2 <= y1:
        raise TrackingError("Pose bounding box must be finite and have positive dimensions.")
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    size = max(x2 - x1, y2 - y1) * 1.25
    transform = np.array([
        [256 / size, 0, 128 - center[0] * 256 / size],
        [0, 256 / size, 128 - center[1] * 256 / size],
    ], np.float32)
    crop = cv2.warpAffine(frame, transform, (256, 256), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - np.array([123.675, 116.28, 103.53], np.float32)) / np.array(
        [58.395, 57.12, 57.375], np.float32
    )
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), center, size


def decode_pose(simcc_x, simcc_y, center, size):
    if simcc_x.shape != (1, 17, 512) or simcc_y.shape != (1, 17, 512):
        raise TrackingError("RTMPose must output AP-10K SimCC tensors of shape (1,17,512).")
    x, y = simcc_x[0], simcc_y[0]
    points = np.stack((x.argmax(1), y.argmax(1)), axis=1).astype(np.float32) / 2
    points = points / 256 * size + center - size / 2
    # Original SimCC confidence convention: raw axis maxima, not softmax probabilities.
    scores = np.minimum(x.max(1), y.max(1))
    return points, scores


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class PoseModel:
    def __init__(self, device="cuda:0", directory=None):
        self.session = ModelSession(
            model_path("rtmpose_ap10k.onnx", directory), (1, 3, 256, 256),
            ((1, 17, 512), (1, 17, 512)), ("simcc_x", "simcc_y"), device,
        )
        self.info = self.session.info

    def predict(self, frame, box):
        tensor, center, size = preprocess_pose(frame, box)
        x, y = self.session.run(tensor)
        return decode_pose(x, y, center, size)


class DogDetector:
    def __init__(self, device="cuda:0", directory=None, threshold=0.3):
        self.session = ModelSession(
            model_path("yolox_tiny.onnx", directory), (1, 3, 416, 416),
            ((1, 3549, 85),), ("output",), device,
        )
        self.info = self.session.info
        self.threshold = threshold
        grids, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[:416 // stride, :416 // stride]
            grids.append(np.stack((x, y), -1).reshape(-1, 2))
            strides.extend([stride] * x.size)
        self.grid = np.concatenate(grids)
        self.strides = np.asarray(strides, np.float32)[:, None]

    def detect(self, frame):
        height, width = frame.shape[:2]
        ratio = min(416 / width, 416 / height)
        resized = cv2.resize(frame, (max(1, int(width * ratio)), max(1, int(height * ratio))))
        canvas = np.full((416, 416, 3), 114, np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        # YOLOX non-legacy ONNX input is BGR, 0..255, CHW; do not normalize it like pose.
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        return self.decode(self.session.run(tensor)[0], ratio, width, height)

    def decode(self, output, ratio, width, height):
        predictions = output[0]
        scores = predictions[:, 4] * predictions[:, 21]  # objectness * COCO dog class 16
        selected = np.flatnonzero(
            (scores >= self.threshold) & (predictions[:, 5:].argmax(1) == 16)
            & np.isfinite(predictions).all(1)
        )
        centers = (predictions[selected, :2] + self.grid[selected]) * self.strides[selected] / ratio
        sizes = np.exp(np.clip(predictions[selected, 2:4], -20, 20)) * self.strides[selected] / ratio
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
        candidates = sorted(
            ((tuple(map(float, b)), float(scores[i])) for b, i in zip(boxes, selected)
             if b[2] - b[0] >= 8 and b[3] - b[1] >= 8),
            key=lambda item: item[1], reverse=True,
        )
        kept = []
        for box, score in candidates:
            if all(box_iou(box, other) < 0.45 for other, _ in kept):
                kept.append((box, score))
        return kept
