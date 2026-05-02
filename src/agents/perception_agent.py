"""
Perception Agent — Image Analysis Module
=========================================
The Perception Agent is the "eyes" of the agentic system. Before any OCR runs,
it inspects the input image and produces a structured ImageProfile describing
brightness, contrast, blur level, skew angle, resolution, and estimated
handwriting density.

This profile drives downstream decisions (which OCR engine to pick, what
preprocessing to apply) instead of using a static, one-size-fits-all pipeline.

Phase 2 Requirement Mapping:
  - Slide 20 (Agentic Concept: Perception)
  - Slide 23 (Agent Architecture: Input → Processing)
  - Slide 25 (Operational Workflow: Observe)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class ImageProfile:
    """Structured result of the perception analysis."""

    filepath: str = ""
    width: int = 0
    height: int = 0
    brightness: float = 0.0          # 0–255 mean luminance
    contrast: float = 0.0            # std-dev of luminance
    blur_score: float = 0.0          # Laplacian variance (low = blurry)
    skew_angle: float = 0.0          # estimated rotation in degrees
    is_low_resolution: bool = False  # width < 1500
    is_dark: bool = False            # brightness < 90
    is_low_contrast: bool = False    # contrast < 40
    is_blurry: bool = False          # blur_score < 50
    is_skewed: bool = False          # abs(skew_angle) > 1.0 degree
    density: str = "normal"          # "sparse", "normal", "dense"
    dominant_color: str = "grayscale"  # "grayscale", "color", "red_ink"
    quality_score: float = 0.0       # 0–100 composite quality rating
    recommendations: list = field(default_factory=list)


class PerceptionAgent:
    """
    Analyzes an input image and returns an ImageProfile.

    Unlike Phase 1 — which blindly applied the same preprocessing regardless
    of image characteristics — the Perception Agent *observes* the image first
    and produces an actionable profile that the Decision Engine uses to adapt
    the pipeline.
    """

    # ---------- thresholds (tunable via memory/feedback) ----------
    BRIGHTNESS_LOW = 90
    BRIGHTNESS_HIGH = 200
    CONTRAST_LOW = 40
    BLUR_THRESHOLD = 50
    SKEW_THRESHOLD = 1.0        # degrees
    LOW_RES_WIDTH = 1500
    DENSE_TEXT_RATIO = 0.35     # ratio of non-zero pixels after binarization
    SPARSE_TEXT_RATIO = 0.08

    def analyze(self, image_path: Path | str) -> ImageProfile:
        """Run full perception analysis on an image file."""
        path = Path(image_path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        profile = ImageProfile(filepath=str(path))
        profile.height, profile.width = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---- brightness & contrast ----
        profile.brightness = float(np.mean(gray))
        profile.contrast = float(np.std(gray))
        profile.is_dark = profile.brightness < self.BRIGHTNESS_LOW
        profile.is_low_contrast = profile.contrast < self.CONTRAST_LOW

        # ---- blur detection (Laplacian variance) ----
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        profile.blur_score = float(laplacian.var())
        profile.is_blurry = profile.blur_score < self.BLUR_THRESHOLD

        # ---- skew estimation ----
        profile.skew_angle = self._estimate_skew(gray)
        profile.is_skewed = abs(profile.skew_angle) > self.SKEW_THRESHOLD

        # ---- resolution ----
        profile.is_low_resolution = profile.width < self.LOW_RES_WIDTH

        # ---- text density ----
        profile.density = self._estimate_density(gray)

        # ---- dominant color / ink detection ----
        profile.dominant_color = self._detect_color_profile(img)

        # ---- composite quality score ----
        profile.quality_score = self._compute_quality_score(profile)

        # ---- generate recommendations ----
        profile.recommendations = self._generate_recommendations(profile)

        return profile

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _estimate_skew(self, gray: np.ndarray) -> float:
        """Estimate document skew via min-area-rect on non-zero pixels."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 12,
        )
        coords = cv2.findNonZero(binary)
        if coords is None or len(coords) < 100:
            return 0.0

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        return round(angle, 2) if abs(angle) > 0.2 else 0.0

    def _estimate_density(self, gray: np.ndarray) -> str:
        """Classify text density as sparse/normal/dense."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 12,
        )
        ratio = np.count_nonzero(binary) / max(1, binary.size)
        if ratio > self.DENSE_TEXT_RATIO:
            return "dense"
        elif ratio < self.SPARSE_TEXT_RATIO:
            return "sparse"
        return "normal"

    def _detect_color_profile(self, img: np.ndarray) -> str:
        """Detect whether the image has significant color ink (e.g. red)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red detection (two hue ranges)
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_ratio = (np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / max(1, img.shape[0] * img.shape[1])

        if red_ratio > 0.01:
            return "red_ink"

        # Check overall saturation
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        if mean_saturation > 30:
            return "color"

        return "grayscale"

    def _compute_quality_score(self, profile: ImageProfile) -> float:
        """
        Compute a 0–100 composite quality score.
        Higher = better quality image for OCR.
        """
        score = 100.0

        # Penalize low brightness
        if profile.is_dark:
            score -= 20

        # Penalize low contrast
        if profile.is_low_contrast:
            score -= 15

        # Penalize blur
        if profile.is_blurry:
            score -= 25

        # Penalize skew
        if profile.is_skewed:
            score -= 10

        # Penalize low resolution
        if profile.is_low_resolution:
            score -= 15

        # Penalize dense text (harder for OCR)
        if profile.density == "dense":
            score -= 10

        return max(0.0, round(score, 1))

    def _generate_recommendations(self, profile: ImageProfile) -> list:
        """
        Generate human-readable recommendations based on the analysis.
        These are shown to the user and also consumed by the Decision Engine.
        """
        recs = []

        if profile.is_dark:
            recs.append("Image is too dark — brightness enhancement recommended")
        if profile.is_low_contrast:
            recs.append("Low contrast detected — CLAHE enhancement recommended")
        if profile.is_blurry:
            recs.append("Image appears blurry — sharpening filter recommended")
        if profile.is_skewed:
            recs.append(f"Document is skewed by {profile.skew_angle:.1f}° — deskew recommended")
        if profile.is_low_resolution:
            recs.append("Low resolution image — upscaling recommended")
        if profile.density == "dense":
            recs.append("Dense text detected — line-by-line OCR strategy recommended")
        if profile.dominant_color == "red_ink":
            recs.append("Red ink detected — dual-channel OCR recommended")

        if not recs:
            recs.append("Image quality is good — standard processing recommended")

        return recs
