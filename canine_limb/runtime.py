"""ONNX Runtime CUDA verification and an explicit OpenCV DNN CPU path."""

from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import os
import sys
import tempfile

import numpy as np


class TrackingError(RuntimeError):
    """An actionable input, runtime, or output error."""


def parse_device(value):
    if value == "cpu":
        return "cpu", 0
    if value == "cuda":
        return "cuda", 0
    if value.startswith("cuda:") and value[5:].isascii() and value[5:].isdigit():
        return "cuda", int(value[5:])
    raise ValueError("Device must be cpu, cuda, or cuda:N (for example cuda:0).")


def cuda_error(reason):
    return TrackingError(
        f"CUDA could not be verified: {reason}. "
        'Use a fresh environment and install with: python -m pip install ".[cuda]". '
        "Check the NVIDIA driver; on Windows also install the Visual C++ x64 runtime. "
        "Then run: python track_limb.py --check-cuda. See README.md."
    )


def _import_ort(cuda):
    if cuda:
        installed = []
        for package in ("onnxruntime", "onnxruntime-gpu"):
            try:
                version(package)
                installed.append(package)
            except PackageNotFoundError:
                pass
        if installed != ["onnxruntime-gpu"]:
            raise cuda_error(
                f"installed ONNX Runtime packages are {installed}; "
                "install only onnxruntime-gpu, not the CPU wheel alongside it"
            )
    try:
        if sys.platform.startswith("linux") and not Path("/proc/cpuinfo").is_file():
            raise TrackingError("ONNX Runtime requires /proc/cpuinfo on Linux; this container does not expose it.")
        import onnxruntime as ort
    except ImportError as exc:
        raise TrackingError(
            'ONNX Runtime is missing. Install ".[cuda]" or, for explicit CPU use, ".[cpu]".'
        ) from exc
    return ort


def verify_cuda(session, device_id):
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise cuda_error(f"the model session uses {providers}")
    actual = session.get_provider_options().get("CUDAExecutionProvider", {}).get("device_id")
    if str(actual) != str(device_id):
        raise cuda_error(f"requested GPU {device_id}, session reports GPU {actual}")


def cuda_profile_counts(profile_path):
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    counts = Counter()
    compute = 0
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider") if event.get("cat") == "Node" else None
        if provider:
            counts[provider] += 1
        if provider == "CUDAExecutionProvider" and args.get("op_name") in {
            "Conv", "FusedConv", "NhwcConv", "MatMul", "FusedMatMul", "Gemm"
        }:
            compute += 1
    if not compute:
        raise cuda_error("warm-up executed no convolution or matrix multiplication on CUDA")
    return dict(counts), compute


class ModelSession:
    def __init__(self, path, input_shape, output_shapes, output_names, device):
        backend, self.device_id = parse_device(device)
        self.cuda = backend == "cuda"
        self.output_names = list(output_names)
        self.output_shapes = output_shapes
        self.input_shape = input_shape
        if not self.cuda:
            # OpenCV is already needed for video I/O and works without a second ORT wheel.
            import cv2
            try:
                self.net = cv2.dnn.readNetFromONNX(str(path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.info = {"device": "cpu", "opencv": cv2.__version__,
                             "cuda_verified": False, "providers": ["OpenCV_CPU"]}
                self.run(np.zeros(input_shape, np.float32))
                return
            except Exception as exc:
                raise TrackingError(f"Could not run CPU ONNX model {Path(path).name}: {exc}") from exc
        ort = _import_ort(self.cuda)
        self.info = {"device": device, "onnxruntime": ort.__version__, "cuda_verified": False}
        try:
            if self.cuda and hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            if self.cuda and "CUDAExecutionProvider" not in ort.get_available_providers():
                raise cuda_error("CUDAExecutionProvider is unavailable")
            with tempfile.TemporaryDirectory(prefix="canine-limb-warmup-") as temp:
                options = ort.SessionOptions()
                options.intra_op_num_threads = min(4, os.cpu_count() or 1)
                options.log_severity_level = 3
                options.enable_profiling = self.cuda
                options.profile_file_prefix = str(Path(temp) / "profile")
                providers = (
                    [("CUDAExecutionProvider", {"device_id": self.device_id}), "CPUExecutionProvider"]
                    if self.cuda else ["CPUExecutionProvider"]
                )
                self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
                self.session.disable_fallback()
                try:
                    inputs = self.session.get_inputs()
                    if len(inputs) != 1 or inputs[0].shape[1:] != list(input_shape[1:]):
                        raise TrackingError(f"Unexpected model input; expected {input_shape}.")
                    self.input_name = inputs[0].name
                    self.run(np.zeros(input_shape, np.float32))
                finally:
                    profile_path = self.session.end_profiling() if self.cuda else None
                if self.cuda:
                    counts, compute = cuda_profile_counts(profile_path)
                    self.info.update(cuda_verified=True, cuda_compute_ops=compute, warmup_providers=counts)
                self.info["providers"] = list(self.session.get_providers())
        except Exception as exc:
            if isinstance(exc, TrackingError):
                raise
            if self.cuda:
                raise cuda_error(str(exc)) from exc
            raise TrackingError(f"Could not run ONNX model {Path(path).name}: {exc}") from exc

    def run(self, tensor):
        if self.cuda:
            verify_cuda(self.session, self.device_id)
        try:
            if self.cuda:
                values = self.session.run(self.output_names, {self.input_name: tensor})
            else:
                self.net.setInput(tensor)
                values = self.net.forward(self.output_names)
        except Exception as exc:
            if self.cuda:
                raise cuda_error(f"inference failed: {exc}") from exc
            raise TrackingError(f"ONNX inference failed: {exc}") from exc
        if len(values) != len(self.output_shapes) or any(
            v.shape != expected or not np.isfinite(v).all()
            for v, expected in zip(values, self.output_shapes)
        ):
            raise TrackingError("The model returned unexpected shapes or non-finite values.")
        return values
