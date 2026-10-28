import numpy as np
import pytest

from canine_limb.models import DogDetector, box_iou, decode_pose, preprocess_pose
from canine_limb.pipeline import frame_rows
from canine_limb.runtime import TrackingError, parse_device
from canine_limb.schema import LIMBS
from canine_limb.tracking import DogTracker


def test_original_animal_keypoint_mapping():
    assert [v[0] for v in LIMBS.values()] == [(5, 6, 7), (8, 9, 10), (11, 12, 13), (14, 15, 16)]
    assert LIMBS["left-hind"][1] == ("hip", "knee", "hind_paw")


def test_pose_rgb_normalization_and_affine_inverse():
    frame = np.full((200, 400, 3), (10, 20, 30), np.uint8)
    tensor, center, size = preprocess_pose(frame, (100, 50, 300, 150))
    assert tensor.shape == (1, 3, 256, 256) and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 128, 128],
                               (np.array([30, 20, 10]) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375], rtol=1e-6)
    x = np.zeros((1, 17, 512), np.float32)
    y = x.copy()
    x[:, :, 256] = 0.7
    y[:, :, 128] = 0.9
    points, scores = decode_pose(x, y, center, size)
    np.testing.assert_allclose(points, np.tile([200, 37.5], (17, 1)))
    np.testing.assert_allclose(scores, 0.7)


def test_bad_pose_crop_and_output_rejected():
    with pytest.raises(TrackingError):
        preprocess_pose(np.zeros((10, 10, 3), np.uint8), (2, 2, 1, 1))
    with pytest.raises(TrackingError):
        decode_pose(np.zeros((1, 17, 384)), np.zeros((1, 17, 512)), np.zeros(2), 100)


def test_missing_rows_and_bounds_never_export_coordinates():
    points = np.full((17, 2), 50.0)
    scores = np.ones(17)
    scores[11] = 0.1
    points[12] = [-1, 50]
    points[13] = [float("nan"), 50]
    rows = frame_rows(2, 20, "left-hind", points, scores, 100, 100, 0.3)
    assert [r["status"] for r in rows] == ["low_confidence", "out_of_frame", "invalid_prediction"]
    assert all(r["x_px"] is None and r["y_px"] is None for r in rows)
    assert rows[0]["score"] == 0.1 and rows[0]["time_s"] == 0.1
    assert len(frame_rows(0, 20, "all", None, None, 100, 100, 0.3)) == 12


def test_detector_decodes_dog_class_with_nms():
    detector = DogDetector.__new__(DogDetector)
    detector.threshold = 0.3
    grids, strides = [], []
    for stride in (8, 16, 32):
        y, x = np.mgrid[:416 // stride, :416 // stride]
        grids.append(np.stack((x, y), -1).reshape(-1, 2))
        strides.extend([stride] * x.size)
    detector.grid = np.concatenate(grids)
    detector.strides = np.asarray(strides)[:, None]
    output = np.zeros((1, 3549, 85), np.float32)
    for index in (0, 1):
        output[0, index, :4] = [10 - index, 10, np.log(10), np.log(10)]
        output[0, index, 4] = 0.9
        output[0, index, 21] = 0.8
    output[0, 2, 4] = 1
    output[0, 2, 5] = 0.99  # person, not dog
    detections = detector.decode(output, 1, 416, 416)
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0][0], [40, 40, 120, 120], atol=1e-4)
    assert detections[0][1] == pytest.approx(0.72)


def test_tracking_continuity_and_lost_dog():
    tracker = DogTracker()
    small = ((200, 0, 240, 40), 0.99)
    large = ((0, 0, 100, 100), 0.8)
    assert tracker.select([small, large])[0] == large
    nearby = ((10, 0, 110, 100), 0.7)
    assert tracker.select([small, nearby])[0] == nearby
    assert tracker.select([]) == (None, "no_dog")
    assert tracker.select([((1000, 1000, 1100, 1100), 0.99)]) == (None, "target_lost")
    assert tracker.previous == nearby[0]
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_ambiguous_dogs_and_explicit_initial_index():
    dogs = [((0, 0, 100, 100), 0.9), ((3, 0, 103, 100), 0.9)]
    tracker = DogTracker(dog_index=1)
    assert tracker.select(dogs)[0] == dogs[1]
    assert tracker.select(dogs) == (None, "ambiguous_dogs")
    assert DogTracker(dog_index=2).select(dogs) == (None, "target_not_found")


@pytest.mark.parametrize("value", ["cuda:-1", "cuda:1.5", "gpu", "cuda:abc", "cuda:"])
def test_invalid_devices(value):
    with pytest.raises(ValueError):
        parse_device(value)


def test_devices():
    assert parse_device("cuda") == ("cuda", 0)
    assert parse_device("cuda:2") == ("cuda", 2)
    assert parse_device("cpu") == ("cpu", 0)
