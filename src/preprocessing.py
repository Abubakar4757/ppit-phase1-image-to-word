from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def _load_bgr(image_path: Path | str) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def _upscale_if_small(img: np.ndarray, min_width: int = 2000) -> np.ndarray:
    if img.shape[1] >= min_width:
        return img
    scale = min_width / float(img.shape[1])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _estimate_skew_angle(gray_or_binary: np.ndarray) -> float:
    if len(gray_or_binary.shape) != 2:
        return 0.0

    if gray_or_binary.dtype != np.uint8:
        image = gray_or_binary.astype(np.uint8)
    else:
        image = gray_or_binary

    inverted = cv2.bitwise_not(image)
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.2:
        return 0.0
    return float(angle)


def _deskew(gray_or_binary: np.ndarray) -> np.ndarray:
    if len(gray_or_binary.shape) != 2:
        return gray_or_binary

    if gray_or_binary.dtype != np.uint8:
        image = gray_or_binary.astype(np.uint8)
    else:
        image = gray_or_binary.copy()

    angle = _estimate_skew_angle(image)
    if angle == 0.0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_for_ocr(image_path: Path | str) -> np.ndarray:
    return preprocess(image_path)


def preprocess(image_path: Path | str) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    if w < 2000:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_for_easyocr(image_path: Path | str) -> np.ndarray:
    bgr = _load_bgr(image_path)
    bgr = _upscale_if_small(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)

    if angle != 0.0:
        h, w = bgr.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        bgr = cv2.warpAffine(
            bgr,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return bgr


def preprocess_with_red_extraction(image_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_img = cv2.bitwise_and(original, original, mask=red_mask)

    main_img = preprocess(image_path)

    return main_img, red_img


def save_preprocessing_stages(
    image_path: Path | str,
    output_dir: Path | str,
) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bgr = _load_bgr(image_path)
    bgr = _upscale_if_small(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )
    deskewed = _deskew(binary)
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(deskewed, cv2.MORPH_CLOSE, kernel)

    files = {
        "input": output_path / "01_input.png",
        "gray": output_path / "02_gray.png",
        "enhanced": output_path / "03_enhanced.png",
        "binary": output_path / "04_binary.png",
        "deskewed": output_path / "05_deskewed.png",
        "final": output_path / "06_final.png",
    }

    cv2.imwrite(str(files["input"]), bgr)
    cv2.imwrite(str(files["gray"]), gray)
    cv2.imwrite(str(files["enhanced"]), enhanced)
    cv2.imwrite(str(files["binary"]), binary)
    cv2.imwrite(str(files["deskewed"]), deskewed)
    cv2.imwrite(str(files["final"]), cleaned)

    return files

