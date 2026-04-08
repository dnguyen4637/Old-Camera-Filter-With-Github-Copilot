# Old Camera Filter — Retro CCD Aesthetic

A Python/OpenCV pipeline that transforms modern, clinical-looking photos and
videos into the moody, organic output of a **late-2000s CCD-sensor digital
camera** (Fujifilm FinePix / Canon PowerShot era).

The goal is to undo the over-sharpened, hyper-detailed look of a modern
smartphone and replace it with something that *feels* more like a physical
photograph — less information, more mood.

> This repository is part of an experiment comparing GitHub Copilot and
> Google's AI coding agents on the same creative brief.

---

## Effects implemented

| Effect | Module function | What it does |
|---|---|---|
| **Halation / Bloom** | `apply_halation()` | Isolates highlights above a luminance threshold and blurs them into a dreamy glow — the CCD's inability to contain bright light becomes a feature. |
| **Crushed Blacks** | `crush_blacks()` | Applies a power-curve toe that pulls shadow values toward pure black, narrowing the dynamic range and focusing the eye on light sources. |
| **Film Grain** | `apply_film_grain()` | Overlays a zero-centred Gaussian noise map on the luminance channel so it reads as silver-halide texture, not digital sensor noise. |
| **Split-tone Color Grade** | `color_grade()` | Shifts mid-tones toward warm cream and deep shadows toward cool teal/blue — the dual-zone colour signature of early digicam JPEG engines. |

---

## Project structure

```
old_camera_filter.py    # Core filter logic — one function per effect
apply_filter.py         # CLI entry point for images and videos
test_old_camera_filter.py  # pytest unit tests (25 tests, all passing)
requirements.txt        # Python dependencies
```

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Apply the filter to an image
python apply_filter.py photo.jpg output.jpg

# Apply the filter to a video
python apply_filter.py clip.mp4 output.mp4

# Override individual effect strengths
python apply_filter.py photo.jpg output.jpg \
    --halation-threshold 0.85 \
    --halation-intensity 0.5  \
    --grain 0.05              \
    --midtone-warmth 0.07     \
    --shadow-teal 0.06
```

Run `python apply_filter.py --help` for a full list of options.

---

## Running the tests

```bash
python -m pytest test_old_camera_filter.py -v
```

---

## Design philosophy

> "Less detail is better."

CCD sensors from the 2000s had limited dynamic range and imprecise colour
science — but those *limitations* produced images that feel intimate and
immediate in a way that modern computational photography does not.  Every
effect in this pipeline deliberately **removes** information:

* Halation spreads highlight detail into a soft glow.
* Black crushing eliminates shadow texture.
* Luminance grain masks fine structure in the mid-tones.
* Split-toning unifies the palette so the image reads as a mood rather
  than a document.

The result is an image that privileges atmosphere over accuracy.
