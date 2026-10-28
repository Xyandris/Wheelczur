import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video(tmp_path):
    def create(count=4, size=(320, 240), fps=20, name="input.avi"):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        assert writer.isOpened(), "MJPEG encoder is required for these tests"
        for i in range(count):
            frame = np.full((size[1], size[0], 3), 40 + i * 3, np.uint8)
            cv2.rectangle(frame, (60 + i, 50), (240 + i, 170), (90, 140, 180), -1)
            writer.write(frame)
        writer.release()
        return path
    return create


@pytest.fixture
def fake_models(monkeypatch):
    from canine_limb import pipeline

    class Pose:
        info = {"device": "test", "cuda_verified": False}

        def __init__(self, *args):
            self.calls = 0

        def predict(self, frame, box):
            self.calls += 1
            points = np.full((17, 2), 100, np.float32)
            scores = np.full(17, 0.9, np.float32)
            for indices, _ in pipeline.LIMBS.values():
                points[list(indices)] = [(120, 80), (140, 120), (160, 160)]
            if self.calls == 2:
                scores[12] = 0.1
            return points, scores

    class Detector:
        info = {"device": "test", "cuda_verified": False}
        sequence = None

        def __init__(self, *args):
            self.calls = 0

        def detect(self, frame):
            if self.sequence is not None:
                result = self.sequence[min(self.calls, len(self.sequence) - 1)]
            else:
                result = [((50, 40, 270, 200), 0.95)]
            self.calls += 1
            return result

    monkeypatch.setattr(pipeline, "PoseModel", Pose)
    monkeypatch.setattr(pipeline, "DogDetector", Detector)
    return Pose, Detector
