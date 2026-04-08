"""
Retro CCD Camera Filter
=======================
Mimics the output of a late-2000s CCD-sensor digital camera (Fujifilm / Canon
PowerShot era).  Four modular stages can be used individually or combined
through the main ``apply_ccd_filter`` pipeline.

All functions operate on float32 BGR images with values in [0, 1].
Use ``load`` / ``save`` for I/O convenience.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load(path: str) -> np.ndarray:
    """Load an image and return a float32 BGR array in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img.astype(np.float32) / 255.0


def save(path: str, img: np.ndarray) -> None:
    """Save a float32 BGR [0, 1] array as an 8-bit image."""
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


# ---------------------------------------------------------------------------
# Stage 1 – Softness & Halation
# ---------------------------------------------------------------------------

def apply_halation(
    img: np.ndarray,
    threshold: float = 0.9,
    blur_radius: int = 31,
    intensity: float = 0.35,
) -> np.ndarray:
    """Add a dreamy glow around the brightest highlights.

    CCD sensors bleed charge from over-exposed pixels into neighbours,
    producing a soft halo rather than hard clipping.  We replicate this by
    isolating high-luminance regions, blurring them heavily, and blending the
    bloom back over the image with a screen-mode composite.

    Less detail is better here: the blur intentionally erases edge information
    in bright areas, trading sharpness for atmosphere.

    Args:
        img: float32 BGR image in [0, 1].
        threshold: Luminance value above which pixels contribute to the bloom.
        blur_radius: Kernel size for the Gaussian blur (must be odd).
        intensity: How strongly the bloom is mixed into the result (0–1).

    Returns:
        float32 BGR image in [0, 1].
    """
    blur_radius = blur_radius | 1  # ensure odd

    # Convert to luminance (perceptual weights)
    gray = (
        0.2126 * img[:, :, 2]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 0]
    )

    # Soft mask: sigmoid-shaped transition above threshold avoids hard edges
    mask = np.clip((gray - threshold) / (1.0 - threshold + 1e-6), 0, 1)
    mask = mask[:, :, np.newaxis]  # (H, W, 1)

    # Isolate and blur the highlight layer
    highlight = img * mask
    bloom = cv2.GaussianBlur(highlight, (blur_radius, blur_radius), 0)

    # Screen blend: result = 1 - (1-base)*(1-blend)
    # This brightens without blowing out already-bright regions.
    screened = 1.0 - (1.0 - img) * (1.0 - bloom * intensity)

    return np.clip(screened, 0, 1)


# ---------------------------------------------------------------------------
# Stage 2 – Subtractive Color & Crushed Blacks
# ---------------------------------------------------------------------------

def crush_blacks(
    img: np.ndarray,
    crush_point: float = 0.18,
    shoulder: float = 0.75,
) -> np.ndarray:
    """Compress the dynamic range by letting shadows fall into deep black.

    A vintage CCD never lifted its blacks for 'clarity'.  Its small sensor
    simply ran out of photons in dim areas and recorded near-zero values.
    We simulate this with a soft power curve that pulls shadows toward zero
    while leaving mid-tones and highlights largely untouched.

    Why 'less detail is better': compressing shadow information removes the
    clean, flat look of digital shadow recovery.  The resulting images feel
    light-centric — the viewer's eye is guided toward illuminated areas.

    Args:
        img: float32 BGR image in [0, 1].
        crush_point: Input luminance below which shadows are heavily compressed.
        shoulder: Input luminance above which highlights are preserved linearly.

    Returns:
        float32 BGR image in [0, 1].
    """
    # Build a smooth S-curve lookup table (256 entries → float32)
    lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    # Shadow region: power > 1 pulls mid-darks toward black
    shadow_gamma = 1.6
    lut = np.where(
        lut < crush_point,
        lut ** shadow_gamma / (crush_point ** (shadow_gamma - 1)),
        lut,
    )

    # Slight highlight roll-off above shoulder for a gentle filmic feel
    lut = np.where(
        lut > shoulder,
        shoulder + (lut - shoulder) * 0.85,
        lut,
    )

    lut = np.clip(lut, 0, 1)

    # Apply per-channel (colour bias in crushed shadows is intentional)
    def _apply_lut(channel: np.ndarray) -> np.ndarray:
        idx = np.clip((channel * 255).astype(np.int32), 0, 255)
        return lut[idx]

    result = np.stack(
        [_apply_lut(img[:, :, c]) for c in range(img.shape[2])],
        axis=2,
    )
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 3 – Film Grain Texture
# ---------------------------------------------------------------------------

def add_film_grain(
    img: np.ndarray,
    intensity: float = 0.045,
    grain_size: float = 1.4,
) -> np.ndarray:
    """Overlay organic film-like grain over the image.

    Digital sensor noise is spatially uniform and mathematically random.
    Physical film grain is clumped — silver-halide crystals vary in size and
    cluster together.  We approximate this by generating Gaussian noise at a
    slightly reduced resolution and upscaling it, which creates the subtle
    'clumping' of real grain.

    Grain is applied in luminosity space so it textures the image without
    shifting colour, matching the monochromatic grain of panchromatic film.

    Why 'less detail is better': a thin layer of grain breaks up smooth digital
    gradients, making flat areas feel tactile rather than clinical.

    Args:
        img: float32 BGR image in [0, 1].
        intensity: Standard deviation of the grain distribution.
        grain_size: Scale factor — values > 1 produce larger, more visible clumps.

    Returns:
        float32 BGR image in [0, 1].
    """
    h, w = img.shape[:2]

    # Generate noise at reduced resolution to simulate grain clustering
    small_h = max(1, int(h / grain_size))
    small_w = max(1, int(w / grain_size))
    noise_small = np.random.normal(0, intensity, (small_h, small_w)).astype(np.float32)
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to luminance, add grain, reconstruct
    # (adding the same noise to all channels keeps grain monochromatic)
    result = img + noise[:, :, np.newaxis]
    return np.clip(result, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 4 – Split-Tone Color Shift
# ---------------------------------------------------------------------------

def apply_color_shift(
    img: np.ndarray,
    midtone_warmth: float = 0.06,
    shadow_teal: float = 0.07,
) -> np.ndarray:
    """Shift midtones toward warm cream and deep shadows toward cool teal.

    Late-2000s Fujifilm Velvia / Provia-influenced processing had a
    characteristic split-tone look: warm, slightly yellow-orange highlights
    and midtones contrasted against blue-shifted shadow areas.  This is a
    direct consequence of the sensor's limited colour matrix and the in-camera
    JPEG engine's aggressive colour science.

    We achieve this with luminance-weighted additive blending:
      - Shadows (low luminance) receive a teal/blue push.
      - Midtones (mid luminance) receive a warm cream push.
      - Highlights are left largely neutral to preserve bloom from Stage 1.

    Args:
        img: float32 BGR image in [0, 1].
        midtone_warmth: Strength of the warm (red/yellow) push in midtones.
        shadow_teal: Strength of the teal (blue/green) push in shadows.

    Returns:
        float32 BGR image in [0, 1].
    """
    gray = (
        0.2126 * img[:, :, 2]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 0]
    )

    # Shadow mask: strong near 0, fades out by ~0.35
    shadow_mask = np.clip(1.0 - gray / 0.35, 0, 1) ** 1.5

    # Midtone mask: bell-shaped, peaks around 0.45 luminance
    mid_mask = np.exp(-((gray - 0.45) ** 2) / (2 * 0.18 ** 2))

    shadow_mask = shadow_mask[:, :, np.newaxis]
    mid_mask = mid_mask[:, :, np.newaxis]

    result = img.copy()

    # Teal push in shadows: lift blue and green, pull red
    # BGR order: [B, G, R]
    teal_color = np.array([shadow_teal * 1.2, shadow_teal * 0.8, -shadow_teal * 0.4],
                          dtype=np.float32)
    result += shadow_mask * teal_color

    # Warm cream push in midtones: lift red and green slightly, leave blue
    warm_color = np.array([-midtone_warmth * 0.1, midtone_warmth * 0.5, midtone_warmth],
                          dtype=np.float32)
    result += mid_mask * warm_color

    return np.clip(result, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def apply_ccd_filter(
    img: np.ndarray,
    halation_intensity: float = 0.35,
    crush_point: float = 0.18,
    grain_intensity: float = 0.045,
    midtone_warmth: float = 0.06,
    shadow_teal: float = 0.07,
) -> np.ndarray:
    """Apply the full retro CCD pipeline to a float32 BGR [0, 1] image.

    Pipeline order matters:
      1. Halation  – applied to the clean input so bloom is not grain-affected.
      2. Crush     – shapes the tonal range before colour grading.
      3. Color     – graded after crush so the split-tone maps to crushed tones.
      4. Grain     – last, so it sits on top of the final colour grade as it
                     would on a physical print.

    Args:
        img: float32 BGR image in [0, 1].
        halation_intensity: Bloom blend strength (Stage 1).
        crush_point: Shadow compression knee (Stage 2).
        grain_intensity: Grain standard deviation (Stage 4).
        midtone_warmth: Warm midtone push strength (Stage 3).
        shadow_teal: Cool shadow push strength (Stage 3).

    Returns:
        float32 BGR image in [0, 1].
    """
    img = apply_halation(img, intensity=halation_intensity)
    img = crush_blacks(img, crush_point=crush_point)
    img = apply_color_shift(img, midtone_warmth=midtone_warmth, shadow_teal=shadow_teal)
    img = add_film_grain(img, intensity=grain_intensity)
    return img
