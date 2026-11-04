#!/usr/bin/env python3
"""Przygotuj środowisko Wheelczur bez dodatkowych narzędzi do instalacji."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent
MODELS = ("rtmpose_ap10k.onnx", "yolox_tiny.onnx")


def command_text(arguments):
    """Komenda do wklejenia w terminalu POSIX lub Windows PowerShell."""
    arguments = [str(value) for value in arguments]
    if os.name == "nt":
        return "& " + " ".join("'" + value.replace("'", "''") + "'" for value in arguments)
    return shlex.join(arguments)


def run(arguments):
    print("\n> " + command_text(arguments), flush=True)
    subprocess.run([str(value) for value in arguments], cwd=ROOT, check=True)


def prepare_environment(directory):
    if directory.exists() and not (directory / "pyvenv.cfg").is_file():
        if not directory.is_dir() or any(directory.iterdir()):
            raise ValueError(
                f"{directory} nie jest środowiskiem venv i nie jest pustym katalogiem. "
                "Wybierz inną ścieżkę przez --venv."
            )
    if not (directory / "pyvenv.cfg").is_file():
        print(f"Tworzenie środowiska: {directory}", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    else:
        print(f"Używanie istniejącego środowiska: {directory}", flush=True)

    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise ValueError(f"Brak interpretera {python}. Wybierz nowy katalog przez --venv.")
    run([python, "-c", (
        "import sys; "
        "sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else "
        "'Środowisko wymaga Pythona 3.10–3.13. Wybierz nowy katalog przez --venv.')"
    )])
    return python


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Zależności dla procesora CPU (domyślnie) lub NVIDIA CUDA.")
    parser.add_argument("--dev", action="store_true", help="Zainstaluj też narzędzia testowe.")
    parser.add_argument("--venv", type=Path, default=Path(".venv"),
                        help="Katalog środowiska; ścieżka względna liczy się od folderu projektu.")
    args = parser.parse_args(argv)

    try:
        if not (3, 10) <= sys.version_info[:2] < (3, 14):
            raise ValueError("Uruchom skrypt Pythonem w wersji od 3.10 do 3.13.")
        missing = [name for name in MODELS if not (ROOT / "canine_limb" / "models" / name).is_file()]
        if missing:
            raise ValueError("Brak modeli w canine_limb/models: " + ", ".join(missing)
                             + ". Rozpakuj pełne archiwum projektu.")

        directory = args.venv.expanduser()
        if not directory.is_absolute():
            directory = ROOT / directory
        python = prepare_environment(directory.resolve())
        run([python, "-m", "pip", "install", "--upgrade", "pip"])
        extras = args.device + (",test" if args.dev else "")
        run([python, "-m", "pip", "install", "-e", f"{ROOT}[{extras}]"])
        run([python, "-m", "pip", "check"])
        run([python, "-c", "import cv2, numpy; from canine_limb.cli import main; "
             "print('Import aplikacji i zależności: OK')"])
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"\nSetup nie powiódł się: {exc}", file=sys.stderr)
        print("Sprawdź komunikat powyżej i połączenie z internetem. "
              "Jeśli brakuje modułu venv/ensurepip, zainstaluj go dla swojego Pythona. "
              "Potem uruchom skrypt ponownie.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nPrzerwano setup. Możesz uruchomić skrypt ponownie.", file=sys.stderr)
        return 130

    print("\nGotowe. Nie trzeba ręcznie aktywować środowiska.")
    if os.name == "nt":
        print("Poniższe komendy są przeznaczone do Windows PowerShell.")
    print("Uruchom analizę (podmień film.mp4 na ścieżkę do nagrania):")
    print(command_text([python, ROOT / "track_limb.py", "film.mp4", "--limb", "left-hind",
                        "--device", "cuda:0" if args.device == "cuda" else "cpu"]))
    if args.device == "cuda":
        print("CUDA wymaga karty NVIDIA i jej sterownika; w Windows także Visual C++ x64 Runtime.")
        print("Sprawdź działanie modeli na GPU:")
        print(command_text([python, ROOT / "track_limb.py", "--check-cuda", "--device", "cuda:0"]))
    if args.dev:
        print("Testy (uruchom w folderze projektu):")
        print(command_text([python, "-m", "pytest"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
