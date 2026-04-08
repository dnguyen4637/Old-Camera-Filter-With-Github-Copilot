"""
demo.py – CLI entry point for the retro CCD camera filter.

Usage
-----
Single image:
    python demo.py input.jpg output.jpg

Video (frame-by-frame processing):
    python demo.py input.mp4 output.mp4

All filter parameters can be tuned via command-line flags.  Run with --help
for the full list.
"""

import argparse
import sys
import os

import cv2
import numpy as np

from filter import apply_ccd_filter, load, save


def _process_image(args: argparse.Namespace) -> None:
    img = load(args.input)
    result = apply_ccd_filter(
        img,
        halation_intensity=args.halation,
        crush_point=args.crush,
        grain_intensity=args.grain,
        midtone_warmth=args.warmth,
        shadow_teal=args.teal,
    )
    save(args.output, result)
    print(f"Saved: {args.output}")


def _process_video(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.input}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.astype(np.float32) / 255.0
        result = apply_ccd_filter(
            img,
            halation_intensity=args.halation,
            crush_point=args.crush,
            grain_intensity=args.grain,
            midtone_warmth=args.warmth,
            shadow_teal=args.teal,
        )
        out_frame = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        out.write(out_frame)

        frame_idx += 1
        if total > 0:
            pct = frame_idx / total * 100
            print(f"\rProcessing frame {frame_idx}/{total} ({pct:.1f}%)", end="", flush=True)

    cap.release()
    out.release()
    print(f"\nSaved: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retro CCD Camera Filter – makes modern footage look like a late-2000s digital camera."
    )
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("output", help="Output image or video path")

    parser.add_argument(
        "--halation", type=float, default=0.35,
        help="Highlight bloom intensity (default: 0.35)",
    )
    parser.add_argument(
        "--crush", type=float, default=0.18,
        help="Shadow crush point / knee (default: 0.18)",
    )
    parser.add_argument(
        "--grain", type=float, default=0.045,
        help="Film grain intensity (default: 0.045)",
    )
    parser.add_argument(
        "--warmth", type=float, default=0.06,
        help="Midtone warm cream push (default: 0.06)",
    )
    parser.add_argument(
        "--teal", type=float, default=0.07,
        help="Shadow teal/blue push (default: 0.07)",
    )

    args = parser.parse_args()

    ext = os.path.splitext(args.output)[1].lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if ext in video_exts:
        _process_video(args)
    else:
        _process_image(args)


if __name__ == "__main__":
    main()
