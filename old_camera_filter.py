"""
old_camera_filter.py
--------------------
Retro CCD camera filter that mimics the output of a late-2000s digital
camera (Fujifilm FinePix / Canon PowerShot era).

Each effect is a self-contained function so the pipeline is easy to
adjust or extend.  All functions operate on float32 images in [0, 1].
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual effect functions
# ---------------------------------------------------------------------------

def apply_halation(frame: np.ndarray,
                   threshold: float = 0.90,
                   blur_radius: int = 31,
                   intensity: float = 0.35) -> np.ndarray:
    """
    Softness & Halation — dreamy glow around bright light sources.

    Isolates pixels whose luminance exceeds *threshold*, blurs them with
    a large Gaussian kernel, then additively blends the bloom back onto the
    frame.  The result is the characteristic 'halation ring' you see on
    CCD sensors when they clip — less detail in the highlights, more mood.

    Parameters
    ----------
    frame      : float32 BGR image in [0, 1]
    threshold  : luminance cutoff above which pixels contribute to the bloom
    blur_radius: Gaussian kernel half-size (must be odd); larger = softer glow
    intensity  : how strongly the bloom is mixed back (0 = none, 1 = full)
    """
    # Ensure the kernel size is odd and at least 1
    ksize = max(1, blur_radius | 1)

    # Compute per-pixel luminance (BT.601 weights on BGR channels)
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    luminance = 0.114 * b + 0.587 * g + 0.299 * r  # (H, W)

    # Isolate bright regions and blur them into a soft bloom map
    highlight_mask = np.clip((luminance - threshold) / (1.0 - threshold + 1e-6), 0, 1)
    highlight_rgb = frame * highlight_mask[:, :, np.newaxis]
    bloom = cv2.GaussianBlur(highlight_rgb, (ksize, ksize), sigmaX=ksize * 0.3)

    return np.clip(frame + bloom * intensity, 0, 1)


def crush_blacks(frame: np.ndarray,
                 toe_start: float = 0.10,
                 toe_strength: float = 1.6) -> np.ndarray:
    """
    Subtractive Color & 'Crushed' Blacks — moody, low-lift shadow rendering.

    Applies a power-curve / toe to the shadow region so that anything below
    *toe_start* falls toward true black faster than a linear ramp would.
    This is the opposite of the lifted-matte look — we want deep blacks that
    swallow detail in the shadows and keep the eye on the light sources.

    Parameters
    ----------
    frame       : float32 BGR image in [0, 1]
    toe_start   : shadow brightness below which the crush begins (0–1)
    toe_strength: gamma exponent applied to the toe region; >1 darkens it
    """
    # Blend between the original and a power-darkened version
    # in the shadow region only, fading smoothly toward toe_start.
    crushed = np.power(np.clip(frame / (toe_start + 1e-6), 0, 1), toe_strength) * toe_start
    shadow_weight = np.clip(1.0 - frame / (toe_start + 1e-6), 0, 1)
    result = frame * (1.0 - shadow_weight) + crushed * shadow_weight
    return np.clip(result, 0, 1)


def apply_film_grain(frame: np.ndarray,
                     intensity: float = 0.035,
                     seed: int | None = None) -> np.ndarray:
    """
    Film Grain Texture — organic, physical-feeling grain.

    Generates luminance-channel noise (not per-channel RGB noise, which
    would look like sensor noise) and adds it uniformly to the frame.
    Using a single luminance grain keeps it feeling like silver-halide
    texture rather than digital sensor noise — less information, more
    character.

    Parameters
    ----------
    frame     : float32 BGR image in [0, 1]
    intensity : peak amplitude of the grain (0 = none)
    seed      : optional RNG seed for reproducible output
    """
    rng = np.random.default_rng(seed)
    h, w = frame.shape[:2]
    # Gaussian grain, zero-centred — same luminance grain on all channels
    grain = rng.normal(loc=0.0, scale=intensity, size=(h, w)).astype(np.float32)
    return np.clip(frame + grain[:, :, np.newaxis], 0, 1)


def color_grade(frame: np.ndarray,
                midtone_warmth: float = 0.05,
                shadow_teal: float = 0.04) -> np.ndarray:
    """
    Color Shift — warm cream mid-tones, cool teal/blue deep shadows.

    Uses a simple split-toning approach:
      • Mid-tones (roughly luminance 0.3–0.8) are pushed toward warm cream
        by lifting red and green slightly while leaving blue untouched.
      • Deep shadows (luminance < 0.3) are nudged toward cool teal by
        lifting blue and green faintly while suppressing red.

    This two-zone shift gives the image the slightly melancholic, golden-
    hour-but-indoors character of early digicam JPEG engines.

    Parameters
    ----------
    frame          : float32 BGR image in [0, 1]
    midtone_warmth : strength of the warm push in mid-tones (0 = none)
    shadow_teal    : strength of the cool push in shadows (0 = none)
    """
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    luminance = 0.114 * b + 0.587 * g + 0.299 * r

    # ── Mid-tone weight: bell centred around 0.55 ────────────────────────
    mid_weight = np.exp(-((luminance - 0.55) ** 2) / (2 * 0.18 ** 2))

    # ── Shadow weight: linear below 0.30, zero above ─────────────────────
    shadow_weight = np.clip(1.0 - luminance / 0.30, 0, 1)

    # Apply warm cream tint to mid-tones (lift R + G, not B)
    r_out = np.clip(r + mid_weight * midtone_warmth, 0, 1)
    g_out = np.clip(g + mid_weight * midtone_warmth * 0.6, 0, 1)
    b_out = b.copy()

    # Apply teal / blue tint to shadows (lift B + G, suppress R)
    r_out = np.clip(r_out - shadow_weight * shadow_teal, 0, 1)
    g_out = np.clip(g_out + shadow_weight * shadow_teal * 0.5, 0, 1)
    b_out = np.clip(b_out + shadow_weight * shadow_teal, 0, 1)

    return np.stack([b_out, g_out, r_out], axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

def apply_ccd_filter(frame_bgr: np.ndarray,
                     halation_threshold: float = 0.90,
                     halation_blur: int = 31,
                     halation_intensity: float = 0.35,
                     toe_start: float = 0.10,
                     toe_strength: float = 1.6,
                     grain_intensity: float = 0.035,
                     midtone_warmth: float = 0.05,
                     shadow_teal: float = 0.04) -> np.ndarray:
    """
    Full retro CCD filter pipeline.

    Accepts and returns a uint8 BGR image (as produced/consumed by OpenCV).
    Internally all processing is done in float32 [0, 1].

    Effect order matters:
      1. Color grade first — establishes the palette before any luminance
         operations touch the values.
      2. Black crush — compresses shadows before the bloom so we don't
         accidentally bloom crushed-black pixels.
      3. Halation — bloom on the already-graded, crushed image.
      4. Grain — added last so it sits on top of the bloom, not under it.
    """
    # ── uint8 → float32 [0, 1] ───────────────────────────────────────────
    img = frame_bgr.astype(np.float32) / 255.0

    img = color_grade(img,
                      midtone_warmth=midtone_warmth,
                      shadow_teal=shadow_teal)

    img = crush_blacks(img,
                       toe_start=toe_start,
                       toe_strength=toe_strength)

    img = apply_halation(img,
                         threshold=halation_threshold,
                         blur_radius=halation_blur,
                         intensity=halation_intensity)

    img = apply_film_grain(img, intensity=grain_intensity)

    # ── float32 [0, 1] → uint8 ───────────────────────────────────────────
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
