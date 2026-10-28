from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from canine_limb.cli import main
from canine_limb.models import DogDetector, PoseModel, model_path
from canine_limb.pipeline import Config, run_video
from canine_limb.runtime import TrackingError


def test_bundled_weights_run_on_cpu():
    cap = cv2.VideoCapture(str(Path(__file__).parent / "data/synthetic_walk.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok
    height, width = frame.shape[:2]
    pose = PoseModel("cpu")
    points, scores = pose.predict(frame, (0, 0, width, height))
    assert points.shape == (17, 2) and scores.shape == (17,)
    assert np.isfinite(points).all() and np.isfinite(scores).all()
    assert pose.info["providers"] == ["OpenCV_CPU"]
    detector = DogDetector("cpu")
    assert isinstance(detector.detect(frame), list)


def test_real_models_end_to_end(make_video, tmp_path):
    video = make_video(count=2)
    automatic = run_video(Config(video, "left-hind", output_dir=tmp_path / "real_auto", device="cpu"))
    fixed = run_video(Config(video, "right-fore", output_dir=tmp_path / "real_fixed", device="cpu",
                             bbox=(0, 0, 320, 240)))
    assert automatic["frames"] == fixed["frames"] == 2
    assert automatic["detector_runtime"]["providers"] == ["OpenCV_CPU"]
    assert fixed["joint_rows"] == 6 and fixed["detector_runtime"] is None


def test_corrupt_weights_rejected(tmp_path):
    (tmp_path / "rtmpose_ap10k.onnx").write_bytes(b"bad model")
    with pytest.raises(TrackingError, match="Checksum mismatch"):
        model_path("rtmpose_ap10k.onnx", tmp_path)


def test_cli_prompts_and_paths_with_spaces(make_video, fake_models, monkeypatch, tmp_path, capsys):
    video = make_video(name="walk with spaces.avi")
    answers = iter([f'"{video}"', "left_hind"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert main(["--device", "cpu", "--output-dir", str(tmp_path / "prompt results")]) == 0
    assert "Finished: 4 frames" in capsys.readouterr().out


def test_cli_noninteractive_alias_and_missing_input(make_video, fake_models, tmp_path, monkeypatch):
    assert main([str(make_video()), "--limb", "right_hind", "--device", "cpu",
                 "--output-dir", str(tmp_path / "cli")]) == 0
    def eof(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert main([]) == 2


def test_cli_help_and_validation():
    result = subprocess.run([sys.executable, "-m", "canine_limb", "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "--check-cuda" in result.stdout
    assert main(["--check-cuda", "--device", "cpu"]) == 1
    with pytest.raises(SystemExit) as exc:
        main(["walk.mp4", "--limb", "left-hind", "--device", "cuda:-1"])
    assert exc.value.code == 2
