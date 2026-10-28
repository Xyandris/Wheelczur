from dataclasses import replace
from pathlib import Path
import csv
import json

import cv2
import numpy as np
import pytest

from canine_limb import pipeline
from canine_limb.pipeline import Config, draw_overlay, frame_rows, run_video
from canine_limb.runtime import TrackingError


def read_rows(result):
    with Path(result["outputs"]["csv"]).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_csv_video_and_metadata_match(make_video, fake_models, tmp_path):
    result = run_video(Config(make_video(), "left-hind", output_dir=tmp_path / "results", device="cpu"))
    rows = read_rows(result)
    assert len(rows) == 12
    assert [int(r["frame"]) for r in rows] == [i for i in range(4) for _ in range(3)]
    assert float(rows[-1]["time_s"]) == pytest.approx(0.15)
    assert rows[4]["status"] == "low_confidence" and rows[4]["x_px"] == ""
    assert result["joint_status_counts"] == {"detected": 11, "low_confidence": 1}
    assert result["frames"] == 4 and result["fps"] == 20
    assert json.loads(Path(result["outputs"]["metadata"]).read_text())["frames"] == 4
    cap = cv2.VideoCapture(result["outputs"]["video"])
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (240, 320, 3)
        count += 1
    cap.release()
    assert count == 4


def test_no_stale_coordinates_when_target_disappears(make_video, fake_models, tmp_path):
    _, detector = fake_models
    detector.sequence = [[((50, 40, 270, 200), 0.95)], [], [((52, 40, 272, 200), 0.95)], []]
    result = run_video(Config(make_video(), "all", output_dir=tmp_path / "missing", device="cpu"))
    rows = read_rows(result)
    assert len(rows) == 48 and result["frames_without_target"] == 2
    missing = [r for r in rows if int(r["frame"]) in {1, 3}]
    assert all(r["status"] == "no_dog" and r["x_px"] == r["y_px"] == r["bbox_x1"] == "" for r in missing)


def test_fixed_bbox_bypasses_detector_and_fps_override(make_video, fake_models, monkeypatch, tmp_path):
    def unexpected(*args):
        raise AssertionError("Detector must not be initialized for a fixed crop")
    monkeypatch.setattr(pipeline, "DogDetector", unexpected)
    result = run_video(Config(make_video(), "right-fore", output_dir=tmp_path / "fixed", device="cpu",
                              bbox=(0, 0, 320, 240), fps=40, video_format="avi"))
    assert result["fps"] == 40 and result["detector_runtime"] is None
    assert float(read_rows(result)[-1]["time_s"]) == pytest.approx(0.075)
    assert read_rows(result)[0]["dog_score"] == ""


def test_existing_outputs_protected(make_video, fake_models, tmp_path):
    config = Config(make_video(), "left-hind", output_dir=tmp_path / "protected", device="cpu")
    result = run_video(config)
    original = Path(result["outputs"]["csv"]).read_bytes()
    with pytest.raises(TrackingError, match="already exists"):
        run_video(config)
    assert Path(result["outputs"]["csv"]).read_bytes() == original
    assert run_video(replace(config, overwrite=True))["frames"] == 4


def test_inference_failure_cleans_temporary_files_and_preserves_existing(make_video, fake_models, monkeypatch, tmp_path):
    pose, _ = fake_models
    config = Config(make_video(), "left-hind", output_dir=tmp_path / "failure", device="cpu")
    result = run_video(config)
    originals = {path: Path(path).read_bytes() for path in result["outputs"].values()}
    def fail(*args):
        raise TrackingError("simulated GPU error")
    monkeypatch.setattr(pose, "predict", fail)
    with pytest.raises(TrackingError, match="simulated GPU error"):
        run_video(replace(config, overwrite=True))
    assert not list(config.output_dir.glob(".canine-limb-*"))
    for path, content in originals.items():
        assert Path(path).read_bytes() == content


def test_interrupt_cleans_open_video_writer(make_video, fake_models, monkeypatch, tmp_path):
    pose, _ = fake_models
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(pose, "predict", interrupt)
    output = tmp_path / "interrupt"
    with pytest.raises(KeyboardInterrupt):
        run_video(Config(make_video(), "left-hind", output_dir=output, device="cpu"))
    assert not list(output.iterdir())


def test_renderer_draws_only_valid_bones():
    image = np.zeros((240, 320, 3), np.uint8)
    points, scores = np.full((17, 2), 60.0), np.ones(17)
    points[[11, 12, 13]] = [(60, 60), (100, 100), (140, 140)]
    rows = frame_rows(0, 20, "left-hind", points, scores, 320, 240, 0.3)
    overlay = draw_overlay(image, rows, None, 0, 20, "left-hind", "detected")
    assert overlay[80, 80].any() and overlay[120, 120].any()
    scores[12] = 0.1
    rows = frame_rows(0, 20, "left-hind", points, scores, 320, 240, 0.3)
    overlay = draw_overlay(image, rows, None, 0, 20, "left-hind", "detected")
    assert not overlay[80, 80].any() and not overlay[120, 120].any()
    assert not image.any()


def test_invalid_paths_and_bbox_fail_cleanly(make_video, fake_models, tmp_path):
    with pytest.raises(TrackingError, match="does not exist"):
        run_video(Config(tmp_path / "absent.mp4", "left-hind", device="cpu"))
    with pytest.raises(TrackingError, match="must fit"):
        run_video(Config(make_video(), "left-hind", device="cpu", bbox=(0, 0, 10000, 10000)))


@pytest.mark.parametrize("change", [{"fps": 0}, {"confidence": -1}, {"detector_confidence": 2}, {"dog_index": -1}])
def test_invalid_numeric_settings(make_video, change):
    with pytest.raises(TrackingError):
        run_video(replace(Config(make_video(), "left-hind", device="cpu"), **change))


def test_odd_dimensions_are_padded_without_changing_csv_coordinates(fake_models, monkeypatch, tmp_path):
    video = tmp_path / "odd-source.avi"
    video.touch()
    real_capture = cv2.VideoCapture

    class OddCapture:
        def __init__(self):
            self.count = 0
        def isOpened(self):
            return True
        def read(self):
            self.count += 1
            return (True, np.zeros((241, 321, 3), np.uint8)) if self.count <= 2 else (False, None)
        def get(self, key):
            return 20 if key == cv2.CAP_PROP_FPS else 2
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: OddCapture() if Path(path) == video else real_capture(path))
    result = run_video(Config(video, "left-hind", device="cpu", output_dir=tmp_path / "odd"))
    assert result["input_size"] == [321, 241] and result["output_size"] == [322, 242]
    assert float(read_rows(result)[0]["x_px"]) == 120


def test_unicode_input_filename(make_video, fake_models, tmp_path):
    result = run_video(Config(make_video(name="zażółć walk.avi"), "left-hind", device="cpu",
                              output_dir=tmp_path / "unicode"))
    assert result["frames"] == 4 and Path(result["outputs"]["video"]).is_file()
