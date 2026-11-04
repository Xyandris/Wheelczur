#!/usr/bin/env python3
"""Pobierz oba modele ONNX Wheelczur do canine_limb/models (bez pip)."""

import argparse
from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ROOT / "canine_limb" / "models"
CHUNK_SIZE = 1024 * 1024
ATTEMPTS = 3


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    sha256: str
    size: int
    member: str = ""
    archive_sha256: str = ""


MODELS = (
    Model(
        "rtmpose_ap10k.onnx",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip",
        "1cfd1c86e0d9e5d5f95178bcd95ee9a4e8386a624cd3c57519f27ff58cac7f28",
        54478120,
        member="end2end.onnx",
        archive_sha256="2d75445331cf2f21d6e164430f96ffa765cd874872965ae1736932dda03987f0",
    ),
    Model(
        "yolox_tiny.onnx",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        20219662,
    ),
)


class DownloadError(RuntimeError):
    """Błąd pobierania lub weryfikacji modelu."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_model(path, model):
    return path.is_file() and path.stat().st_size == model.size and sha256(path) == model.sha256


def download(url, destination):
    request = Request(url, headers={"User-Agent": "Wheelczur-model-downloader/1.0"})
    with urlopen(request, timeout=30) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_report = -1
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            done += len(chunk)
            if done > 256 * 1024 * 1024:
                raise DownloadError("Odpowiedź serwera jest większa niż oczekiwany model.")
            output.write(chunk)
            report = int(done / total * 10) if total else done // (4 * CHUNK_SIZE)
            if report != last_report:
                size = f" / {total / CHUNK_SIZE:.1f} MiB" if total else " MiB"
                print(f"  Pobrano {done / CHUNK_SIZE:.1f}{size}", flush=True)
                last_report = report
        if total and done != total:
            raise DownloadError(f"Niepełne pobranie: {done} z {total} bajtów.")


def install_model(model, directory):
    target = directory / model.filename
    if valid_model(target, model):
        print(f"OK: {model.filename} już istnieje i ma poprawną sumę SHA-256.", flush=True)
        return target
    if target.exists() and not target.is_file():
        raise DownloadError(f"{target} nie jest zwykłym plikiem.")
    if target.exists():
        print(f"{model.filename}: plik jest niekompletny lub uszkodzony; pobieram ponownie.", flush=True)

    for attempt in range(1, ATTEMPTS + 1):
        try:
            print(f"Pobieranie {model.filename} (próba {attempt}/{ATTEMPTS})...", flush=True)
            # Pliki tymczasowe są na tym samym dysku; dopiero sprawdzony model zastępuje docelowy plik.
            with tempfile.TemporaryDirectory(prefix=f".{model.filename}-", dir=directory) as temp:
                payload = Path(temp) / "download"
                download(model.url, payload)
                candidate = payload
                if model.member:
                    if sha256(payload) != model.archive_sha256:
                        raise DownloadError("Niepoprawna suma SHA-256 pobranego archiwum.")
                    candidate = Path(temp) / model.filename
                    with ZipFile(payload) as archive:
                        members = [info for info in archive.infolist()
                                   if not info.is_dir() and PurePosixPath(info.filename).name == model.member]
                        if len(members) != 1 or members[0].file_size != model.size:
                            raise DownloadError(f"Archiwum nie zawiera oczekiwanego modelu {model.member}.")
                        # Odczytujemy tylko jeden model; ścieżki wewnątrz ZIP nie wyznaczają miejsca zapisu.
                        with archive.open(members[0]) as src, candidate.open("wb") as dst:
                            shutil.copyfileobj(src, dst, CHUNK_SIZE)
                if not valid_model(candidate, model):
                    raise DownloadError("Niepoprawny rozmiar lub suma SHA-256 modelu.")
                os.replace(candidate, target)
            print(f"Gotowe: {target}", flush=True)
            return target
        except (OSError, HTTPException, BadZipFile, ValueError, DownloadError) as exc:
            if attempt == ATTEMPTS:
                raise DownloadError(
                    f"Nie udało się pobrać {model.filename}: {exc}. "
                    "Sprawdź połączenie z internetem i uruchom skrypt ponownie. "
                    "Poprawnie pobrane modele zostaną pominięte."
                ) from exc
            print(f"  Błąd: {exc}. Ponawiam...", flush=True)
            time.sleep(attempt)


def ensure_models(directory=None):
    directory = Path(directory).expanduser().resolve() if directory is not None else DEFAULT_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Folder modeli: {directory}", flush=True)
    return tuple(install_model(model, directory) for model in MODELS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path,
                        help="Inny folder docelowy; domyślnie canine_limb/models obok skryptu.")
    args = parser.parse_args(argv)
    try:
        ensure_models(args.models_dir)
        print("\nOba modele są gotowe. Możesz uruchomić aplikację.")
        return 0
    except (OSError, DownloadError) as exc:
        print(f"\nBłąd: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nPrzerwano. Uruchom skrypt ponownie, aby dokończyć pobieranie.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    exit_code = main()
    # Przy dwukliku w Windows pozostawiamy okno z wynikiem; setup importuje funkcję bez pauzy.
    if os.name == "nt" and len(sys.argv) == 1 and sys.stdin.isatty():
        try:
            input("\nNaciśnij Enter, aby zamknąć okno...")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(exit_code)
