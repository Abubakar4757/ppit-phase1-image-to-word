from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, List

import pytesseract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_for_easyocr, preprocess_for_ocr


CHEMISTRY_TERMS = [
    "adsorption",
    "absorption",
    "desorption",
    "surface",
    "chemistry",
    "enthalpy",
    "entropy",
    "gibbs",
    "energy",
    "physical",
    "chemical",
    "coagulation",
    "colloidal",
    "catalysis",
    "electrophoresis",
    "turbidity",
    "micelles",
    "lyophilic",
    "lyophobic",
    "emulsion",
    "aerosol",
    "sol",
    "gel",
    "reaction",
    "temperature",
    "pressure",
    "isotherm",
    "freundlich",
    "langmuir",
    "electrolyte",
]


@dataclass
class ResultRow:
    mode: str
    image: str
    chars: int
    lines: int
    duration_sec: float
    output_file: str


def _load_images(images_dir: Path) -> List[Path]:
    images = sorted([
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
        *images_dir.glob("*.png"),
    ])
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def _configure_tesseract() -> None:
    found = shutil.which("tesseract")
    if not found:
        default_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default_path.exists():
            found = str(default_path)
    if found:
        pytesseract.pytesseract.tesseract_cmd = found


def _run_tesseract_raw(image_path: Path) -> str:
    _configure_tesseract()
    return pytesseract.image_to_string(str(image_path), config="--oem 1 --psm 6")


def _run_tesseract_preprocessed(image_path: Path) -> str:
    _configure_tesseract()
    processed = preprocess_for_ocr(image_path)
    return pytesseract.image_to_string(processed, config="--oem 1 --psm 6")


_EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def _easyocr_to_text(results: list) -> str:
    extracted = []
    for idx, item in enumerate(results):
        if isinstance(item, str):
            clean = item.strip()
            if clean:
                extracted.append((float(idx * 20), 0.0, 0.0, clean, 1.0))
            continue

        if not isinstance(item, (list, tuple)):
            continue

        if len(item) >= 3:
            box, text, conf = item[0], item[1], float(item[2])
        elif len(item) == 2:
            box, text = item[0], item[1]
            conf = 1.0
        else:
            continue

        if conf < 0.15:
            continue
        clean = text.strip()
        if len(clean) <= 1 and not clean.isalnum():
            continue

        if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)) and len(box[0]) >= 2:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            min_y = float(min(ys))
            min_x = float(min(xs))
            max_x = float(max(xs))
        else:
            min_y = float(idx * 20)
            min_x = 0.0
            max_x = 0.0

        extracted.append((min_y, min_x, max_x, clean, conf))

    if not extracted:
        return ""

    x_values = [item[1] for item in extracted]
    min_x = min(x_values)
    max_x = max(x_values)
    split_x = (min_x + max_x) / 2.0

    has_two_columns = False
    if len(extracted) >= 16 and (max_x - min_x) > 350:
        left_count = sum(1 for item in extracted if item[1] <= split_x)
        right_count = len(extracted) - left_count
        if left_count >= 4 and right_count >= 4:
            has_two_columns = True

    if has_two_columns:
        extracted.sort(key=lambda x: (0 if x[1] <= split_x else 1, x[0], x[1]))
    else:
        extracted.sort(key=lambda x: (x[0], x[1]))

    if not extracted:
        return ""

    lines = []
    current_line = [extracted[0][3]]
    current_y = extracted[0][0]
    current_col = 0 if (has_two_columns and extracted[0][1] <= split_x) else 1

    for y, x0, _x1, text, _conf in extracted[1:]:
        col = 0 if (has_two_columns and x0 <= split_x) else 1

        same_column = col == current_col
        if same_column and abs(y - current_y) <= 18:
            current_line.append(text)
        else:
            lines.append(" ".join(current_line))
            current_line = [text]
            current_y = y
            current_col = col
    lines.append(" ".join(current_line))

    return "\n".join(lines)


def _run_easyocr_raw(image_path: Path) -> str:
    reader = _get_easyocr_reader()
    results = reader.readtext(str(image_path), detail=1, paragraph=False)
    return _easyocr_to_text(results)


def _run_easyocr_preprocessed(image_path: Path) -> str:
    reader = _get_easyocr_reader()
    processed = preprocess_for_easyocr(image_path)
    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        contrast_ths=0.05,
        adjust_contrast=0.6,
        text_threshold=0.55,
        low_text=0.2,
    )
    return _easyocr_to_text(results)


def _count_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


def _best_term_match(token: str) -> str:
    lower = token.lower()
    if len(lower) < 4 or not lower.isalpha():
        return token

    best = lower
    best_score = 0.0
    for term in CHEMISTRY_TERMS:
        score = SequenceMatcher(None, lower, term).ratio()
        if score > best_score:
            best_score = score
            best = term

    if best_score >= 0.78:
        if token[0].isupper():
            return best.capitalize()
        return best
    return token


def clean_ocr_text(text: str) -> str:
    cleaned = text.replace("|", " ")
    cleaned = cleaned.replace("~", " ")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("@", "a")
    cleaned = re.sub(r"\s+", " ", cleaned)

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.replace("|", " ").replace("~", " ").replace("_", " ")
        parts = re.findall(r"[A-Za-z]+|\d+|[^A-Za-z\d\s]", line)
        repaired_parts = [_best_term_match(part) for part in parts]
        rebuilt = " ".join(repaired_parts)
        rebuilt = re.sub(r"\s+([,.;:!?])", r"\1", rebuilt)
        rebuilt = re.sub(r"\s+", " ", rebuilt).strip()
        if rebuilt:
            lines.append(rebuilt)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR quality pass with preprocessing variants")
    parser.add_argument("--images-dir", default="data/sample_images")
    parser.add_argument("--results-dir", default="benchmark/quality_results")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    images_dir = (project_root / args.images_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    images = _load_images(images_dir)

    modes: list[tuple[str, Callable[[Path], str]]] = [
        ("tesseract_raw", _run_tesseract_raw),
        ("tesseract_preprocessed", _run_tesseract_preprocessed),
        ("easyocr_raw", _run_easyocr_raw),
        ("easyocr_cleaned", lambda p: clean_ocr_text(_run_easyocr_raw(p))),
        ("easyocr_preprocessed", _run_easyocr_preprocessed),
    ]

    rows: List[ResultRow] = []

    for image_idx, image_path in enumerate(images, start=1):
        for mode_name, mode_fn in modes:
            started = time.perf_counter()
            try:
                text = mode_fn(image_path)
            except Exception as exc:
                text = f"[ERROR] {exc}"
            duration = time.perf_counter() - started

            out_name = f"{mode_name}_image{image_idx}.txt"
            out_path = results_dir / out_name
            out_path.write_text(text, encoding="utf-8")

            rows.append(
                ResultRow(
                    mode=mode_name,
                    image=image_path.name,
                    chars=len(text),
                    lines=_count_lines(text),
                    duration_sec=duration,
                    output_file=str(out_path.relative_to(project_root)),
                )
            )
            print(f"[DONE] {mode_name} on {image_path.name} ({duration:.2f}s)")

    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "image", "chars", "lines", "duration_sec", "output_file"])
        for row in rows:
            writer.writerow([row.mode, row.image, row.chars, row.lines, f"{row.duration_sec:.4f}", row.output_file])

    md_path = results_dir / "summary.md"
    md_lines = [
        "# OCR Quality Pass Summary",
        "",
        "| Mode | Image | Chars | Lines | Time (s) | Output |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row.mode} | {row.image} | {row.chars} | {row.lines} | {row.duration_sec:.2f} | {row.output_file} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
