"""Streaming video processing, CSV export, and skeleton rendering."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os
import tempfile
import time

import cv2
import numpy as np

from . import __version__
from .models import DogDetector, MODEL_HASHES, PoseModel
from .runtime import TrackingError, parse_device
from .schema import COLORS, CSV_FIELDS, LIMBS, selected_limbs
from .tracking import DogTracker


@dataclass
class Config:
    video: Path
    limb: str
    output_dir: Path | None = None
    device: str = "cuda:0"
    confidence: float = 0.3
    detector_confidence: float = 0.3
    dog_index: int = 0
    bbox: tuple | None = None  # fixed X Y WIDTH HEIGHT, in original pixels
    fps: float | None = None
    video_format: str = "mp4"
    models_dir: Path | None = None
    overwrite: bool = False


def frame_rows(frame_index, fps, limb, points, scores, width, height, threshold,
               box=None, dog_score=None, missing_status="no_dog"):
    rows = []
    for name in selected_limbs(limb):
        indices, joints = LIMBS[name]
        for index, joint in zip(indices, joints):
            x = y = score = None
            status = missing_status
            if points is not None:
                px, py = map(float, points[index])
                raw_score = float(scores[index])
                score = raw_score if math.isfinite(raw_score) else None
                if not all(map(math.isfinite, (px, py, raw_score))):
                    status = "invalid_prediction"
                elif raw_score < threshold:
                    status = "low_confidence"
                elif not (0 <= px < width and 0 <= py < height):
                    status = "out_of_frame"
                else:
                    x, y, status = px, py, "detected"
            rows.append(dict(zip(CSV_FIELDS, (
                frame_index, round(frame_index / fps, 9), name, joint,
                x, y, score, status, dog_score,
                *(box if box is not None else (None,) * 4),
            ))))
    return rows


def draw_overlay(frame, rows, box, frame_index, fps, limb, target_status):
    result = frame.copy()
    height, width = result.shape[:2]
    scale = max(0.35, min(0.7, width / 1000))
    line_width = max(2, round(width / 500))
    radius = max(3, round(width / 250))
    if box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (160, 160, 160), 1)
    for name in selected_limbs(limb):
        joints = [row for row in rows if row["limb"] == name]
        color = COLORS[name]
        for a, b in zip(joints, joints[1:]):
            if a["status"] == b["status"] == "detected":
                cv2.line(result, (round(a["x_px"]), round(a["y_px"])),
                         (round(b["x_px"]), round(b["y_px"])), color, line_width, cv2.LINE_AA)
        for joint in joints:
            if joint["status"] == "detected":
                xy = round(joint["x_px"]), round(joint["y_px"])
                cv2.circle(result, xy, radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(result, xy, radius, color, -1, cv2.LINE_AA)
                label = joint["joint"].replace("_", " ")
                cv2.putText(result, label, (xy[0] + radius + 3, xy[1] - radius),
                            cv2.FONT_HERSHEY_SIMPLEX, scale * 0.85, (15, 15, 15), 3, cv2.LINE_AA)
                cv2.putText(result, label, (xy[0] + radius + 3, xy[1] - radius),
                            cv2.FONT_HERSHEY_SIMPLEX, scale * 0.85, color, 1, cv2.LINE_AA)
    valid = sum(r["status"] == "detected" for r in rows)
    state = f"{valid}/{len(rows)} joints" if box is not None else target_status.replace("_", " ")
    title = f"RTMPose | {limb} | frame {frame_index} | {frame_index / fps:.2f}s | {state}"
    text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    if text_width > width - 16:
        scale *= max(0.1, (width - 16) / text_width)
    cv2.rectangle(result, (0, 0), (width, min(height - 1, 30)), (25, 25, 25), -1)
    cv2.putText(result, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)
    return result


def _validate_config(config):
    parse_device(config.device)
    if config.limb not in (*LIMBS, "all"):
        raise TrackingError(f"Unknown limb: {config.limb}.")
    if not math.isfinite(config.confidence) or config.confidence < 0:
        raise TrackingError("Joint confidence threshold must be finite and nonnegative.")
    if not math.isfinite(config.detector_confidence) or not 0 <= config.detector_confidence <= 1:
        raise TrackingError("Detector confidence threshold must be between 0 and 1.")
    if config.dog_index < 0:
        raise TrackingError("Dog index must be nonnegative.")
    if config.fps is not None and (not math.isfinite(config.fps) or config.fps <= 0):
        raise TrackingError("FPS must be finite and greater than zero.")
    if config.video_format not in {"mp4", "avi"}:
        raise TrackingError("Video format must be mp4 or avi.")


def _verify_output(path, frame_count, width, height):
    check = cv2.VideoCapture(str(path))
    try:
        ok, frame = check.read()
        count = round(check.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or frame is None or frame.shape[:2] != (height, width) or count != frame_count:
            raise TrackingError(
                "The encoded overlay could not be verified. Try --video-format avi "
                "and check free disk space."
            )
    finally:
        check.release()


def run_video(config, progress=None):
    _validate_config(config)
    video = Path(config.video).expanduser().resolve()
    if not video.is_file():
        raise TrackingError(f"Video does not exist: {video}")
    output_dir = (Path(config.output_dir).expanduser().resolve() if config.output_dir
                  else video.parent / f"{video.stem}_tracking")
    prefix = f"{video.stem}_{config.limb.replace('-', '_')}"
    paths = {
        "csv": output_dir / f"{prefix}.csv",
        "video": output_dir / f"{prefix}_overlay.{config.video_format}",
        "metadata": output_dir / f"{prefix}_run.json",
    }
    for path in paths.values():
        if path.exists() and (not config.overwrite or not path.is_file()):
            raise TrackingError(f"Output already exists: {path}. Choose another --output-dir or use --overwrite.")
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        if not cap.isOpened():
            raise TrackingError(f"OpenCV cannot open the video: {video}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise TrackingError("The video contains no decodable frames.")
        height, width = frame.shape[:2]
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = config.fps if config.fps is not None else source_fps
        if not math.isfinite(fps) or fps <= 0:
            raise TrackingError("Video FPS is missing or invalid; supply --fps 30 (or the correct rate).")
        reported_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        expected = max(0, round(reported_count)) if math.isfinite(reported_count) else 0
        fixed_box = None
        if config.bbox is not None:
            if len(config.bbox) != 4 or not np.isfinite(config.bbox).all():
                raise TrackingError("--bbox needs four finite numbers: X Y WIDTH HEIGHT.")
            x, y, bw, bh = config.bbox
            if x < 0 or y < 0 or bw <= 0 or bh <= 0 or x + bw > width or y + bh > height:
                raise TrackingError(f"--bbox must fit inside the {width}x{height} input frame.")
            fixed_box = (x, y, x + bw, y + bh)
        pose = PoseModel(config.device, config.models_dir)
        detector = (DogDetector(config.device, config.models_dir, config.detector_confidence)
                    if fixed_box is None else None)
        tracker = DogTracker(config.dog_index)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Even output dimensions prevent MPEG-4 encoders from dropping an odd row/column.
        encoded_width, encoded_height = width + width % 2, height + height % 2
        started = time.monotonic()
        counts = Counter()
        missing_frames = 0
        with tempfile.TemporaryDirectory(prefix=".canine-limb-", dir=output_dir) as temporary, ExitStack() as cleanup:
            staged = {key: Path(temporary) / path.name for key, path in paths.items()}
            codec = "mp4v" if config.video_format == "mp4" else "MJPG"
            writer = cv2.VideoWriter(str(staged["video"]), cv2.VideoWriter_fourcc(*codec), fps,
                                     (encoded_width, encoded_height))
            cleanup.callback(writer.release)
            if not writer.isOpened():
                raise TrackingError("Could not create overlay video. Try --video-format avi and check folder permissions.")
            processed = 0
            with staged["csv"].open("w", encoding="utf-8-sig", newline="") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                csv_writer.writeheader()
                while ok:
                    if frame is None or frame.shape[:2] != (height, width):
                        raise TrackingError(f"Frame {processed} has changed dimensions or cannot be decoded.")
                    box, dog_score, points, scores = fixed_box, None, None, None
                    target_status = "fixed_bbox" if fixed_box else "no_dog"
                    if detector is not None:
                        detection, target_status = tracker.select(detector.detect(frame))
                        if detection is not None:
                            box, dog_score = detection
                    if box is not None:
                        points, scores = pose.predict(frame, box)
                    else:
                        missing_frames += 1
                    rows = frame_rows(processed, fps, config.limb, points, scores, width, height,
                                      config.confidence, box, dog_score, target_status)
                    csv_writer.writerows(rows)
                    counts.update(row["status"] for row in rows)
                    overlay = draw_overlay(frame, rows, box, processed, fps, config.limb, target_status)
                    if (encoded_height, encoded_width) != (height, width):
                        overlay = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2,
                                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    writer.write(overlay)
                    processed += 1
                    if progress:
                        progress(processed, expected)
                    ok, frame = cap.read()
            writer.release()
            writer = None
            if expected and processed < expected:
                raise TrackingError(
                    f"Decoder stopped after {processed} frames, but the source reports {expected}. "
                    "Repair/transcode the input video before retrying."
                )
            _verify_output(staged["video"], processed, encoded_width, encoded_height)
            metadata = {
                "program_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
                "input": str(video), "limb": config.limb, "device": config.device,
                "frames": processed, "source_reported_frames": expected,
                "source_fps": source_fps if math.isfinite(source_fps) else None,
                "fps": fps, "time_basis": "zero-based frame index / output fps (constant frame rate)",
                "input_size": [width, height], "output_size": [encoded_width, encoded_height],
                "coordinate_system": "original input pixels; origin top-left; x right, y down",
                "confidence_threshold": config.confidence,
                "detector_confidence_threshold": config.detector_confidence,
                "score_definition": "minimum of raw SimCC x/y maxima; not a calibrated probability",
                "dog_index": config.dog_index, "fixed_bbox_xywh": config.bbox,
                "joint_rows": sum(counts.values()), "joint_status_counts": dict(counts),
                "frames_without_target": missing_frames,
                "pose_runtime": pose.info, "detector_runtime": detector.info if detector else None,
                "model_sha256": {name: digest for name, digest in MODEL_HASHES.items()
                                 if detector is not None or name == "rtmpose_ap10k.onnx"},
                "outputs": {key: str(path) for key, path in paths.items()},
                "processing_seconds": round(time.monotonic() - started, 3),
                "codec": codec, "audio": False,
            }
            staged["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            # Finish and verify all files, and re-check collisions, before publishing.
            for destination in paths.values():
                if destination.exists() and not config.overwrite:
                    raise TrackingError(f"Output was created by another process: {destination}")
            for key, destination in paths.items():
                os.replace(staged[key], destination)
        return metadata
    finally:
        cap.release()
        if writer is not None:
            writer.release()
