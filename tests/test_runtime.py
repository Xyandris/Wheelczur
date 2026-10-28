from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np
import pytest

from canine_limb import runtime
from canine_limb.runtime import ModelSession, TrackingError


@pytest.fixture
def fake_ort(monkeypatch):
    state = SimpleNamespace(
        active=["CUDAExecutionProvider", "CPUExecutionProvider"],
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        actual_id=None, fail=False, preloaded=False, sessions=[],
        events=[{"cat": "Node", "args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"}}],
    )

    class Session:
        def __init__(self, path, sess_options, providers):
            self.options, self.providers = sess_options, providers
            self.device_id = providers[0][1]["device_id"]
            self.fallback_disabled = False
            state.sessions.append(self)

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 256, 256])]

        def get_providers(self):
            return state.active

        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"device_id": str(self.device_id if state.actual_id is None else state.actual_id)}}

        def run(self, names, inputs):
            assert self.fallback_disabled
            assert inputs["input"].shape == (1, 3, 256, 256)
            if state.fail:
                raise RuntimeError("simulated device failure")
            return [np.zeros((1, 17, 512), np.float32) for _ in names]

        def end_profiling(self):
            path = self.options.profile_file_prefix + ".json"
            Path(path).write_text(json.dumps(state.events))
            return path

    def preload():
        state.preloaded = True

    ort = SimpleNamespace(__version__="test", SessionOptions=SimpleNamespace,
                          InferenceSession=Session, preload_dlls=preload,
                          get_available_providers=lambda: state.available)
    monkeypatch.setattr(runtime, "_import_ort", lambda cuda: ort)
    return state


def session():
    return ModelSession("unused.onnx", (1, 3, 256, 256), ((1, 17, 512),) * 2,
                        ("simcc_x", "simcc_y"), "cuda:2")


def test_cuda_checks_actual_compute_and_selected_device(fake_ort):
    model = session()
    assert fake_ort.preloaded and fake_ort.sessions[0].fallback_disabled
    assert fake_ort.sessions[0].providers[0] == ("CUDAExecutionProvider", {"device_id": 2})
    assert model.info["cuda_verified"] and model.info["cuda_compute_ops"] == 1
    assert not Path(fake_ort.sessions[0].options.profile_file_prefix).parent.exists()


def test_cuda_unavailable(fake_ort):
    fake_ort.available = ["CPUExecutionProvider"]
    with pytest.raises(TrackingError, match="unavailable"):
        session()


def test_cpu_fallback_rejected(fake_ort):
    fake_ort.active = ["CPUExecutionProvider"]
    with pytest.raises(TrackingError, match="could not be verified"):
        session()


@pytest.mark.parametrize("op", ["MemcpyFromHost", "Shape"])
def test_gpu_registration_or_copy_without_compute_rejected(fake_ort, op):
    fake_ort.events = [{"cat": "Node", "args": {"provider": "CUDAExecutionProvider", "op_name": op}}]
    with pytest.raises(TrackingError, match="no convolution"):
        session()


def test_wrong_gpu_rejected(fake_ort):
    fake_ort.actual_id = 0
    with pytest.raises(TrackingError, match="requested GPU 2"):
        session()


def test_gpu_failure_is_actionable_and_does_not_fallback(fake_ort):
    model = session()
    fake_ort.fail = True
    with pytest.raises(TrackingError, match="simulated device failure"):
        model.run(np.zeros((1, 3, 256, 256), np.float32))
    assert fake_ort.sessions[0].fallback_disabled


def test_provider_change_after_warmup_rejected(fake_ort):
    model = session()
    fake_ort.active = ["CPUExecutionProvider"]
    with pytest.raises(TrackingError, match="could not be verified"):
        model.run(np.zeros((1, 3, 256, 256), np.float32))


def test_cpu_and_gpu_wheels_must_not_coexist(monkeypatch):
    monkeypatch.setattr(runtime, "version", lambda name: "1.23.2")
    with pytest.raises(TrackingError, match="only onnxruntime-gpu"):
        runtime._import_ort(cuda=True)


@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get("CANINE_TEST_CUDA") != "1", reason="Set CANINE_TEST_CUDA=1 on an NVIDIA host")
def test_actual_cuda_models():
    from canine_limb.models import DogDetector, PoseModel
    for model in (PoseModel("cuda:0"), DogDetector("cuda:0")):
        assert model.info["cuda_verified"]
        assert model.info["cuda_compute_ops"] > 0
