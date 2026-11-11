import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

import launcher
from canine_limb import runtime


@pytest.fixture
def windows_cuda(tmp_path, monkeypatch):
    root = tmp_path / "site packages"
    packages = {}
    state = SimpleNamespace(registered={}, loaded={}, calls=[])
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime, "_CUDA_DLL_DIRECTORY_HANDLES", {})
    monkeypatch.setattr(runtime, "_CUDA_EXTRA_DLL_HANDLES", {})
    monkeypatch.setenv("PATH", "existing-bin")

    def install(package, *names):
        files = []
        for name in names:
            relative = Path(name)
            actual = root / relative
            actual.parent.mkdir(parents=True, exist_ok=True)
            actual.touch()
            files.append(relative)
        packages[package] = SimpleNamespace(files=files, locate_file=lambda entry: root / entry)
        return root / files[0].parent

    def distribution(name):
        if name not in packages:
            raise runtime.PackageNotFoundError(name)
        return packages[name]

    def register(path):
        handle = object()
        state.registered[path] = handle
        state.calls.append(("register", path))
        return handle

    def load(path):
        handle = object()
        state.loaded[path] = handle
        state.calls.append(("load", path))
        return handle

    def preload(**kwargs):
        # NVIDIA folders must already be visible when ORT starts loading DLLs.
        for path in state.registered:
            assert path in runtime.os.environ["PATH"].split(runtime.os.pathsep)
        state.calls.append(("ort", kwargs))

    monkeypatch.setattr(runtime, "distribution", distribution)
    monkeypatch.setattr(runtime.os, "add_dll_directory", register, raising=False)
    monkeypatch.setattr(ctypes, "CDLL", load)
    state.install = install
    state.ort = SimpleNamespace(preload_dlls=preload)
    return state


def test_windows_loads_new_cudnn_engines_before_inference_and_keeps_handles(windows_cuda):
    state = windows_cuda
    cudnn = state.install(
        "nvidia-cudnn-cu12",
        "nvidia/cudnn/bin/cudnn64_9.dll",
        "nvidia/cudnn/bin/cudnn_engines_tensor_ir64_9.dll",
        "nvidia/cudnn/bin/cudnn_ext64_9.dll",
        "nvidia/cudnn/include/cudnn.h",
    )
    nvrtc = state.install("nvidia-cuda-nvrtc-cu12", "nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll")
    runtime.preload_cuda_libraries(state.ort)
    paths = runtime.os.environ["PATH"].split(runtime.os.pathsep)
    assert set(paths[:-1]) == {str(cudnn), str(nvrtc)}
    assert paths[-1] == "existing-bin"
    assert runtime._CUDA_DLL_DIRECTORY_HANDLES == state.registered
    assert runtime._CUDA_EXTRA_DLL_HANDLES == state.loaded
    assert set(state.loaded) == {
        str(cudnn / "cudnn_engines_tensor_ir64_9.dll"), str(cudnn / "cudnn_ext64_9.dll")
    }
    assert state.calls[-3:] == [
        ("ort", {"directory": ""}),
        ("load", str(cudnn / "cudnn_ext64_9.dll")),
        ("load", str(cudnn / "cudnn_engines_tensor_ir64_9.dll")),
    ]
    first_calls = list(state.calls)
    runtime.preload_cuda_libraries(state.ort)
    assert state.calls == first_calls + [("ort", {"directory": ""})]
    assert runtime.os.environ["PATH"].split(runtime.os.pathsep) == paths


def test_older_cudnn_does_not_require_new_optional_engines(windows_cuda):
    state = windows_cuda
    state.install("nvidia-cudnn-cu12", "nvidia/cudnn/bin/cudnn64_9.dll")
    runtime.preload_cuda_libraries(state.ort)
    assert state.calls[-1] == ("ort", {"directory": ""})
    assert not state.loaded


def test_manual_cuda_installation_keeps_standard_search_behavior(windows_cuda):
    state = windows_cuda
    runtime.preload_cuda_libraries(state.ort)
    assert state.calls == [("ort", {})]
    assert runtime.os.environ["PATH"] == "existing-bin"


def test_failed_cudnn_load_identifies_library_and_repair_instructions(windows_cuda, monkeypatch):
    state = windows_cuda
    state.install("nvidia-cudnn-cu12", "nvidia/cudnn/bin/cudnn_engines_tensor_ir64_9.dll")

    def fail(path):
        raise OSError("A dependency could not be loaded")

    monkeypatch.setattr(ctypes, "CDLL", fail)
    with pytest.raises(runtime.TrackingError, match="cudnn_engines_tensor_ir64_9.dll.*repair"):
        runtime.preload_cuda_libraries(state.ort)
    assert not runtime._CUDA_EXTRA_DLL_HANDLES


def test_linux_preserves_existing_loading_and_path(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setenv("PATH", "existing-linux-bin")

    def unexpected(name):
        raise AssertionError("Linux must not register Windows DLL directories")

    monkeypatch.setattr(runtime, "distribution", unexpected)
    runtime.preload_cuda_libraries(SimpleNamespace(preload_dlls=lambda: calls.append("preload")))
    assert calls == ["preload"]
    assert runtime.os.environ["PATH"] == "existing-linux-bin"


def test_launcher_check_uses_same_cuda_loader(monkeypatch, capsys):
    calls = []
    ort = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(runtime, "_import_ort", lambda cuda: ort)
    monkeypatch.setattr(runtime, "preload_cuda_libraries", lambda loaded: calls.append(loaded))
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.sys, "argv", ["check", "cuda"])

    def device_count(pointer):
        pointer._obj.value = 1
        return 0

    driver = SimpleNamespace(cuInit=lambda flags: 0, cuDeviceGetCount=device_count)
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: driver, raising=False)
    exec(launcher.DEPENDENCY_CHECK, {})
    assert calls == [ort]
    assert "NVIDIA GPUs: 1" in capsys.readouterr().out
