#!/usr/bin/env python3
"""
apply_filter.py
---------------
CLI entry point for the retro CCD camera filter.

Usage
-----
# Process a single image:
    python apply_filter.py photo.jpg output.jpg

# Process a video:
    python apply_filter.py clip.mp4 output.mp4

# Override individual effect strengths (see --help for all options):
    python apply_filter.py photo.jpg output.jpg --grain 0.05 --halation-intensity 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2

from old_camera_filter import apply_ccd_filter

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply a retro late-2000s CCD camera filter to an image or video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  help="Path to the input image or video file.")
    p.add_argument("output", help="Path to write the filtered output.")

    # ── Halation ─────────────────────────────────────────────────────────
    p.add_argument("--halation-threshold", type=float, default=0.90,
                   metavar="F",
                   help="Luminance threshold above which pixels contribute to the bloom (0–1).")
    p.add_argument("--halation-blur", type=int, default=31,
                   metavar="N",
                   help="Gaussian blur radius for the bloom (pixels, must be odd).")
    p.add_argument("--halation-intensity", type=float, default=0.35,
                   metavar="F",
                   help="Bloom mix strength (0 = none, 1 = full).")

    # ── Black crush ───────────────────────────────────────────────────────
    p.add_argument("--toe-start", type=float, default=0.10,
                   metavar="F",
                   help="Shadow brightness below which the crush begins (0–1).")
    p.add_argument("--toe-strength", type=float, default=1.6,
                   metavar="F",
                   help="Gamma exponent applied to the shadow toe; >1 darkens it.")

    # ── Grain ─────────────────────────────────────────────────────────────
    p.add_argument("--grain", type=float, default=0.035,
                   metavar="F",
                   dest="grain_intensity",
                   help="Peak amplitude of the film grain (0 = none).")

    # ── Color grade ───────────────────────────────────────────────────────
    p.add_argument("--midtone-warmth", type=float, default=0.05,
                   metavar="F",
                   help="Strength of the warm-cream push in the mid-tones (0 = none).")
    p.add_argument("--shadow-teal", type=float, default=0.04,
                   metavar="F",
                   help="Strength of the cool-teal push in deep shadows (0 = none).")

    return p


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_image(input_path: Path, output_path: Path, **kwargs) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Error: could not read image '{input_path}'.")

    filtered = apply_ccd_filter(img, **kwargs)

    if not cv2.imwrite(str(output_path), filtered):
        sys.exit(f"Error: could not write image to '{output_path}'.")

    print(f"Image saved → {output_path}")


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(input_path: Path, output_path: Path, **kwargs) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"Error: could not open video '{input_path}'.")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Choose a codec that works for the requested extension
    suffix = output_path.suffix.lower()
    fourcc_str = "mp4v" if suffix in {".mp4", ".mov"} else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        sys.exit(f"Error: could not create output video '{output_path}'.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(apply_ccd_filter(frame, **kwargs))
        frame_idx += 1
        if total > 0:
            pct = frame_idx / total * 100
            print(f"\rProcessing frame {frame_idx}/{total} ({pct:.0f}%)", end="", flush=True)

    cap.release()
    out.release()
    print(f"\nVideo saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Error: input file '{input_path}' does not exist.")

    kwargs = dict(
        halation_threshold = args.halation_threshold,
        halation_blur      = args.halation_blur,
        halation_intensity = args.halation_intensity,
        toe_start          = args.toe_start,
        toe_strength       = args.toe_strength,
        grain_intensity    = args.grain_intensity,
        midtone_warmth     = args.midtone_warmth,
        shadow_teal        = args.shadow_teal,
    )

    suffix = input_path.suffix.lower()
    if suffix in IMAGE_EXTS:
        process_image(input_path, output_path, **kwargs)
    elif suffix in VIDEO_EXTS:
        process_video(input_path, output_path, **kwargs)
    else:
        sys.exit(
            f"Error: unrecognised file extension '{suffix}'.\n"
            f"Supported images: {sorted(IMAGE_EXTS)}\n"
            f"Supported videos: {sorted(VIDEO_EXTS)}"
        )


if __name__ == "__main__":
    main()
