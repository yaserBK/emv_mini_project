"""
augment.py -- Image augmentation for bottle cap training data.

Provides randomised augmentation of 256x256 circle-masked cap crops.
Each call to AugParams independently samples every parameter, so the
joint distribution gives dense coverage near neutral with exponentially
thinning probability toward extreme combinations.

Augmentation distributions
--------------------------
  gamma        log-normal  mu=0, sigma=0.30  -> typically 0.55-1.82  (V channel)
  saturation   log-normal  mu=0, sigma=0.28  -> typically 0.60-1.67  (S channel)
  brightness   normal      mu=0, sigma=12    -> +/-~24 additive luma shift
  rotation     disabled    always 0deg
  skew         uniform     +/-8deg          (perspective tilt, two axes)
  noise        half-normal sigma=6          -> 0-~18 std Gaussian pixel noise
  blur         Bernoulli   p=0.25       -> slight Gaussian blur (kernel 3 or 5)
"""

from typing import Tuple

import cv2
import numpy as np

BG_COLOUR: Tuple[int, int, int] = (0, 0, 0)


class AugParams:
    """Draws and stores one complete set of random augmentation parameters."""

    def __init__(self, rng: np.random.Generator):
        # Gamma: log-normal sigma=0.30 -> ~[0.55, 1.82], hard-clipped to [0.4, 2.5]
        self.gamma = float(np.clip(np.exp(rng.normal(0, 0.30)), 0.40, 2.50))

        # Saturation scale: log-normal sigma=0.28 -> ~[0.60, 1.67]
        self.sat_scale = float(np.clip(np.exp(rng.normal(0, 0.28)), 0.40, 2.50))

        # Brightness shift: normal mu=0, sigma=12, additive to V channel
        self.brightness = float(np.clip(rng.normal(0, 12), -40, 40))

        # Rotation: disabled
        self.rotation = 0.0

        # Perspective skew: uniform +/-8deg on each axis
        self.skew_x = float(rng.uniform(-8, 8))
        self.skew_y = float(rng.uniform(-8, 8))

        # Gaussian noise std: half-normal sigma=6 -> always non-negative
        self.noise_std = float(np.abs(rng.normal(0, 6)))

        # Blur: on with probability 0.25
        self.blur = rng.random() < 0.25
        self.blur_k = int(rng.choice([3, 5]))

    def describe(self) -> dict:
        return {
            "gamma":      round(self.gamma, 3),
            "sat_scale":  round(self.sat_scale, 3),
            "brightness": round(self.brightness, 2),
            "rotation":   round(self.rotation, 2),
            "skew_x":     round(self.skew_x, 2),
            "skew_y":     round(self.skew_y, 2),
            "noise_std":  round(self.noise_std, 3),
            "blur":       self.blur,
            "blur_k":     self.blur_k,
        }


def apply_colour(img: np.ndarray, p: AugParams) -> np.ndarray:
    """Apply gamma, saturation scale, and brightness shift in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Gamma on V channel (multiplicative, log-normal distributed)
    hsv[..., 2] = np.clip((hsv[..., 2] / 255.0) ** (1.0 / p.gamma) * 255.0, 0, 255)

    # Saturation scale
    hsv[..., 1] = np.clip(hsv[..., 1] * p.sat_scale, 0, 255)

    # Additive brightness shift
    hsv[..., 2] = np.clip(hsv[..., 2] + p.brightness, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_geometry(
    img: np.ndarray,
    p: AugParams,
    bg: Tuple[int, int, int] = BG_COLOUR,
) -> np.ndarray:
    """Apply rotation then perspective skew, keeping the cap inside the frame."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Rotation
    M_rot = cv2.getRotationMatrix2D((cx, cy), p.rotation, 1.0)
    img = cv2.warpAffine(
        img, M_rot, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )

    # Perspective skew: model camera not perpendicular to cap surface
    dx = np.tan(np.deg2rad(abs(p.skew_x))) * h * 0.5
    dy = np.tan(np.deg2rad(abs(p.skew_y))) * w * 0.5
    sx = np.sign(p.skew_x)
    sy = np.sign(p.skew_y)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [max(0,  sx * dx),      max(0,  sy * dy)],
        [min(w, w - sx * dx),  max(0, -sy * dy)],
        [min(w, w + sx * dx),  min(h, h - sy * dy)],
        [max(0, -sx * dx),     min(h, h + sy * dy)],
    ])

    M_persp = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(
        img, M_persp, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )
    return img


def apply_noise(img: np.ndarray, p: AugParams) -> np.ndarray:
    """Add Gaussian noise; no-op when std is negligible."""
    if p.noise_std < 0.5:
        return img
    noise = np.random.normal(0, p.noise_std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, p: AugParams) -> np.ndarray:
    """Apply a slight Gaussian blur if selected."""
    if not p.blur:
        return img
    k = p.blur_k if p.blur_k % 2 == 1 else p.blur_k + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def augment_crop(crop: np.ndarray, p: AugParams) -> np.ndarray:
    """Apply the full augmentation chain (colour -> geometry -> noise -> blur)."""
    aug = apply_colour(crop, p)
    aug = apply_geometry(aug, p, bg=BG_COLOUR)
    aug = apply_noise(aug, p)
    aug = apply_blur(aug, p)
    return aug
