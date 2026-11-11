#!/usr/bin/env python3
"""Interactive Wheelczur launcher; use --detailed for complete subprocess output."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = {'cpu': '.venv-cpu', 'cuda': '.venv-cuda'}
DETAILED = False
LIMBS = ('left-fore', 'right-fore', 'left-hind', 'right-hind')

# Run by the chosen venv. These checks do not load or download model files.
ENV_CHECK = """
import sys
from pathlib import Path
expected = Path(sys.argv[1]).resolve()
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != expected:
    raise SystemExit('Wrong interpreter: expected the selected virtual environment.')
if not (3, 10) <= sys.version_info[:2] < (3, 14):
    raise SystemExit('Wheelczur requires Python 3.10 through 3.13.')
print('Environment Python:', sys.executable, flush=True)
"""

DEPENDENCY_CHECK = """
import sys
import cv2
import numpy
from canine_limb.cli import main
if sys.argv[1] == 'cuda':
    import ctypes
    from canine_limb.runtime import _import_ort, preload_cuda_libraries
    ort = _import_ort(True)
    preload_cuda_libraries(ort)
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise SystemExit('CUDAExecutionProvider is unavailable. Run GPU setup.')
    driver = ctypes.WinDLL('nvcuda.dll') if sys.platform == 'win32' else ctypes.CDLL('libcuda.so.1')
    code = driver.cuInit(0)
    if code:
        raise SystemExit('NVIDIA driver initialization failed (CUDA error %s).' % code)
    count = ctypes.c_int()
    code = driver.cuDeviceGetCount(ctypes.byref(count))
    if code or count.value < 1:
        raise SystemExit('No usable NVIDIA CUDA GPU found (CUDA error %s).' % code)
    print('CUDA provider available; NVIDIA GPUs:', count.value)
print('Application dependencies imported successfully.')
"""

PACKAGE_INFO = """
import json
from importlib.metadata import PackageNotFoundError, version
installed = []
for name in ('onnxruntime', 'onnxruntime-gpu'):
    try:
        version(name)
        installed.append(name)
    except PackageNotFoundError:
        pass
print(json.dumps(installed))
"""

PREPARE_ENV = """
import sys
from pathlib import Path
from setup_env import prepare_environment
prepare_environment(Path(sys.argv[1]))
"""


def environment(device):
    return ROOT / ENVIRONMENTS[device]


def environment_python(device):
    return environment(device) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def run_command(arguments, *, label=None, tracking=False, timeout=None):
    """Quiet by default, with only tracker progress/results and useful failures."""
    arguments = [str(arg) for arg in arguments]
    if label:
        print(label, flush=True)
    if DETAILED:
        display = subprocess.list2cmdline(arguments) if os.name == 'nt' else shlex.join(arguments)
        print('\n> ' + display, flush=True)
        result = subprocess.run(arguments, cwd=ROOT, check=False, timeout=timeout)
        if result.returncode:
            print(f'Failed (exit {result.returncode}). See output above.', flush=True)
        return result.returncode == 0
    if not tracking:
        result = subprocess.run(arguments, cwd=ROOT, check=False, timeout=timeout,
                                capture_output=True, text=True, errors='replace')
        if result.returncode:
            show_failure((result.stdout or '') + '\n' + (result.stderr or ''), result.returncode)
        return result.returncode == 0

    # Prompts are collected by the launcher so captured child output cannot hide them.
    child_env = os.environ.copy()
    child_env['PYTHONUNBUFFERED'] = '1'
    tail = deque(maxlen=30)
    progress = False
    with subprocess.Popen(arguments, cwd=ROOT, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          errors='replace', bufsize=1, env=child_env) as process:
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                tail.append(line)
                if line.startswith('Processed '):
                    print('\r' + line + '    ', end='', flush=True)
                    progress = True
                elif line.startswith(('Finished:', 'csv:', 'video:', 'metadata:', 'json:', 'No valid joints')):
                    if progress:
                        print()
                        progress = False
                    print(line, flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if progress:
        print()
    if code:
        show_failure('\n'.join(tail), code)
    return code == 0


def show_failure(output, code):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    errors = [line for line in lines if 'error' in line.lower() or 'błąd' in line.lower()]
    summary = errors[-1] if errors else (lines[-1] if lines else 'No error details were returned.')
    print(f'Failed (exit {code}): {summary[:900]}', flush=True)
    print('Use python3.12 launcher.py --detailed for the full output.', flush=True)


def require_project():
    missing = [name for name in ('setup_env.py', 'track_limb.py', 'pyproject.toml',
                                 'download_models.py', 'canine_limb') if not (ROOT / name).exists()]
    if missing:
        print('Put launcher.py beside track_limb.py in the complete Wheelczur project.')
        print('Missing: ' + ', '.join(missing))
        return False
    return True


def require_environment(device):
    directory = environment(device)
    python = environment_python(device)
    if not (directory / 'pyvenv.cfg').is_file() or not python.is_file():
        print(f'{device.upper()} environment is not set up. Choose Setup {device.upper()} first.')
        return False
    return run_command([python, '-c', ENV_CHECK, directory])


def check(device):
    print(f'Checking {device.upper()} environment...', flush=True)
    if not require_environment(device):
        return False
    python = environment_python(device)
    if not run_command([python, '-m', 'pip', 'check']):
        print(f'Choose Setup {device.upper()} to repair dependencies.')
        return False
    if not run_command([python, '-c', DEPENDENCY_CHECK, device]):
        print(f'Check failed. Choose Setup {device.upper()}; for GPU, also check the NVIDIA driver.')
        return False
    print(f'{device.upper()} environment: OK. Models are checked separately.', flush=True)
    if device == 'cuda':
        print('CUDA model execution will be verified when you run tracking.')
    return True


def model_inventory():
    from download_models import MODELS, valid_model
    directory = ROOT / 'canine_limb' / 'models'
    return [(model, valid_model(directory / model.filename, model)) for model in MODELS]


def check_models(download=False):
    print('Checking model files and SHA-256 checksums...', flush=True)
    inventory = model_inventory()
    for model, valid in inventory:
        print(f'  {model.filename}: {"OK" if valid else "missing or invalid"}')
    if all(valid for _, valid in inventory):
        print('Models: ready.')
        return True
    if not download:
        print('Choose Check / download models (option 7) to get the missing or invalid files.')
        return False
    if not run_command([sys.executable, ROOT / 'download_models.py', '--models-dir',
                        ROOT / 'canine_limb' / 'models'],
                       label='Downloading missing or invalid models (internet required)...'):
        return False
    if not all(valid for _, valid in model_inventory()):
        print('Model verification failed. Retry option 7.')
        return False
    print('Models: ready. Checksums verified.')
    return True


def setup(device):
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        print('Start this launcher with Python 3.10 through 3.13 (for example python3.12).')
        return False
    directory = environment(device)
    if not run_command([sys.executable, '-c', PREPARE_ENV, directory],
                       label=f'Preparing {device.upper()} environment...'):
        return False
    if not require_environment(device):
        return False
    python = environment_python(device)
    if device == 'cuda':
        result = subprocess.run([str(python), '-c', PACKAGE_INFO], cwd=ROOT,
                                check=False, capture_output=True, text=True)
        if DETAILED:
            print(result.stdout, end='')
            print(result.stderr, end='', file=sys.stderr)
        if result.returncode:
            show_failure(result.stdout + '\n' + result.stderr, result.returncode)
            return False
        installed = json.loads(result.stdout)
        if 'onnxruntime' in installed:
            if not run_command([python, '-m', 'pip', 'uninstall', '-y', *installed],
                               label='Repairing conflicting ONNX Runtime packages in the GPU environment...'):
                return False
    if not run_command([python, '-m', 'pip', 'install', '--upgrade', 'pip'], label='Updating pip...'):
        return False
    if not run_command([python, '-m', 'pip', 'install', '-e', f'{ROOT}[{device}]'],
                       label=f'Installing {device.upper()} packages (internet required; please wait)...'):
        return False
    return check(device)


def prompt_tracking():
    print('Relative video paths start in:', ROOT)
    value = input('Path to video: ').strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise ValueError('Enter a video path.')
    video = Path(value).expanduser()
    if not video.is_absolute():
        video = ROOT / video
    if not video.is_file():
        raise ValueError(f'Video not found: {video}')
    print("Choose a limb (left/right refer to the dog's own sides):")
    for number, limb in enumerate(LIMBS, 1):
        print(f'  {number}. {limb}')
    while True:
        answer = input('Limb number or name: ').strip().lower().replace('_', '-')
        if answer in LIMBS:
            return video, answer
        if answer in {str(i) for i in range(1, len(LIMBS) + 1)}:
            return video, LIMBS[int(answer) - 1]
        print('Choose 1–4 or enter a listed limb name.')


def run_tracker(device):
    if not check(device) or not check_models():
        return False
    python = environment_python(device)
    if device == 'cuda':
        if not run_command([python, ROOT / 'track_limb.py', '--check-cuda', '--device', 'cuda:0'],
                           label='Verifying actual model computation on GPU...'):
            return False
        print('CUDA model computation: OK.')
    video, limb = prompt_tracking()
    return run_command([python, ROOT / 'track_limb.py', video, '--limb', limb,
                        '--device', 'cuda:0' if device == 'cuda' else 'cpu'],
                       label=f'Starting tracking on {device.upper()}...', tracking=True)


def status():
    for device in ENVIRONMENTS:
        present = (environment(device) / 'pyvenv.cfg').is_file() and environment_python(device).is_file()
        print(f'{device.upper()}: {"environment found; choose Check to verify" if present else "not set up"}')
    if DETAILED:
        print('Project:', ROOT)
        print('Launcher Python:', sys.executable)
        for device in ENVIRONMENTS:
            print(device.upper(), environment_python(device))
        executable = shutil.which('nvidia-smi')
        if executable:
            run_command([executable], timeout=15)


def perform(action, device=None):
    try:
        if action == 'status':
            status()
            return True
        if action == 'models':
            return check_models(download=True)
        return {'setup': setup, 'check': check, 'run': run_tracker}[action](device)
    except (KeyboardInterrupt, EOFError):
        print('\nAction cancelled.')
        return False
    except (OSError, subprocess.SubprocessError, ValueError, ImportError) as exc:
        if DETAILED:
            import traceback
            traceback.print_exc()
        else:
            print(f'Action failed: {exc}')
            print('Use --detailed for the full output.')
        return False


def main(argv=None):
    global DETAILED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detailed', action='store_true', help='Show all setup, check and tracking output')
    actions = parser.add_mutually_exclusive_group()
    for action in ('setup', 'check', 'run'):
        actions.add_argument('--' + action, choices=('cpu', 'cuda'), help=f'{action.capitalize()} without the menu')
    actions.add_argument('--models', action='store_true', help='Verify models and download missing or invalid files')
    actions.add_argument('--status', action='store_true', help='Show environment presence')
    args = parser.parse_args(argv)
    DETAILED = args.detailed
    if not require_project():
        return 1
    for action in ('setup', 'check', 'run'):
        device = getattr(args, action)
        if device:
            return 0 if perform(action, device) else 1
    for action in ('models', 'status'):
        if getattr(args, action):
            return 0 if perform(action) else 1
    options = {'1': ('setup', 'cpu'), '2': ('setup', 'cuda'),
               '3': ('run', 'cpu'), '4': ('run', 'cuda'),
               '5': ('check', 'cpu'), '6': ('check', 'cuda'),
               '7': ('models', None), '8': ('status', None)}
    while True:
        print('\n=== Wheelczur launcher ===' + (' [detailed output]' if DETAILED else '') + '\n'
              '1. Setup / update CPU environment\n'
              '2. Setup / update GPU environment (NVIDIA CUDA)\n'
              '3. Run tracker on CPU\n'
              '4. Run tracker on GPU (CUDA GPU 0)\n'
              '5. Check CPU environment\n'
              '6. Check GPU environment\n'
              '7. Check / download models\n'
              '8. Show environment status\n'
              '0. Exit')
        try:
            answer = input('Choose an option: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye.')
            return 0
        if answer in ('0', 'q', 'quit', 'exit'):
            return 0
        if answer not in options:
            print('Choose a number from 0 to 8.')
            continue
        perform(*options[answer])


if __name__ == '__main__':
    raise SystemExit(main())
