"""Interactive prompts and equivalent non-interactive CLI arguments."""

from pathlib import Path
import argparse
import json
import math
import sys
import time

from . import __version__
from .runtime import TrackingError, parse_device
from .schema import LIMBS


def device_argument(value):
    try:
        parse_device(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def finite_number(value):
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a finite number.") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("Expected a finite number.")
    return result


def parser():
    p = argparse.ArgumentParser(
        description="Track a dog's selected limb with RTMPose and save CSV + skeleton video.",
        epilog='Example: python track_limb.py "walk.mp4" --limb left-hind --device cuda:0',
    )
    p.add_argument("video", nargs="?", help="Input video path; prompts when omitted.")
    p.add_argument("--limb", type=lambda s: s.replace("_", "-"), choices=(*LIMBS, "all"),
                   help="Animal's left/right side; prompts when omitted.")
    p.add_argument("--device", type=device_argument, default="cuda:0", help="cuda:N (default cuda:0), or explicit cpu.")
    p.add_argument("--output-dir", type=Path, help="Default: <video_folder>/<video_stem>_tracking/.")
    p.add_argument("--confidence", type=finite_number, default=0.3, help="Minimum raw SimCC score (default 0.3).")
    p.add_argument("--detector-confidence", type=finite_number, default=0.3, help="Minimum dog detection probability (default 0.3).")
    p.add_argument("--dog-index", type=int, default=0, help="Initial target: 0=largest dog, 1=next largest, etc.")
    p.add_argument("--bbox", type=finite_number, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                   help="Fixed whole-dog crop in pixels; skips automatic detection. Include the whole dog.")
    p.add_argument("--fps", type=finite_number, help="Override FPS when metadata is wrong/missing.")
    p.add_argument("--video-format", choices=("mp4", "avi"), default="mp4", help="mp4=mp4v (default), avi=MJPEG.")
    p.add_argument("--models-dir", type=Path, help="Directory containing the unchanged supplied ONNX models.")
    p.add_argument("--overwrite", action="store_true", help="Replace existing outputs after processing succeeds.")
    p.add_argument("--check-cuda", action="store_true", help="Run both models and verify actual CUDA compute, then exit.")
    p.add_argument("--version", action="version", version=__version__)
    return p


def prompt_limb():
    options = (*LIMBS, "all")
    print("Choose a limb (left/right refer to the dog's own sides):")
    for i, name in enumerate(options, 1):
        print(f"  {i}. {name}")
    while True:
        answer = input("Limb number or name: ").strip().lower().replace("_", "-")
        if answer in options:
            return answer
        if answer in {str(i) for i in range(1, len(options) + 1)}:
            return options[int(answer) - 1]
        print("Choose 1-5 or enter a listed limb name.")


def input_path(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise TrackingError("Enter a video path.")
    return Path(value).expanduser()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.check_cuda:
            if parse_device(args.device)[0] != "cuda":
                raise TrackingError("--check-cuda requires --device cuda:N.")
            from .models import DogDetector, PoseModel
            pose = PoseModel(args.device, args.models_dir)
            detector = DogDetector(args.device, args.models_dir)
            print(json.dumps({"pose": pose.info, "detector": detector.info}, indent=2))
            return 0
        video = input_path(args.video if args.video is not None else input("Path to video: "))
        limb = args.limb or prompt_limb()
        from .pipeline import Config, run_video
        last_update = 0.0

        def progress(done, total):
            nonlocal last_update
            now = time.monotonic()
            if now - last_update >= 1 or done == total:
                count = f"{done}/{total}" if total else str(done)
                print(f"\rProcessed {count} frames", end="", file=sys.stderr, flush=True)
                last_update = now

        print(f"Using {args.device}; loading RTMPose and {'fixed crop' if args.bbox else 'dog detector'}...", file=sys.stderr)
        result = run_video(Config(
            video=video, limb=limb, output_dir=args.output_dir, device=args.device,
            confidence=args.confidence, detector_confidence=args.detector_confidence,
            dog_index=args.dog_index, bbox=tuple(args.bbox) if args.bbox else None,
            fps=args.fps, video_format=args.video_format, models_dir=args.models_dir,
            overwrite=args.overwrite,
        ), progress=progress)
        print(file=sys.stderr)
        print(f"Finished: {result['frames']} frames; {result['joint_status_counts'].get('detected', 0)} valid joint observations.")
        for label, path in result["outputs"].items():
            print(f"{label}: {path}")
        if not result["joint_status_counts"].get("detected", 0):
            print("No valid joints were detected. Check the footage/crop and the confidence threshold.", file=sys.stderr)
        return 0
    except EOFError:
        print("Error: input ended. Supply VIDEO and --limb for non-interactive use.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Unfinished temporary outputs are discarded.", file=sys.stderr)
        return 130
    except (TrackingError, OSError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f'Error: missing dependency ({exc}). Install with: python -m pip install ".[cuda]"', file=sys.stderr)
        return 1
