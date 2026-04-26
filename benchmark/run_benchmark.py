
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List


@dataclass
class OCRResult:
    engine: str
    image_name: str
    output_file: str
    duration_sec: float
    status: str
    message: str


def _load_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = sorted(
        [
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
            *images_dir.glob("*.png"),
        ]
    )
    if not images:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return images


def run_tesseract(img_path: Path, config: str = "") -> str:
    from PIL import Image
    import pytesseract

    configured_path = shutil.which("tesseract")
    if not configured_path:
        default_windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default_windows_path.exists():
            configured_path = str(default_windows_path)
    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = configured_path

    image = Image.open(img_path)
    return pytesseract.image_to_string(image, config=config)


def run_easyocr(img_path: Path) -> str:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(str(img_path), detail=1)
    lines = [r[1] for r in results]
    return "\n".join(lines)


def run_paddleocr(img_path: Path) -> str:
    from paddleocr import PaddleOCR

    os.environ.setdefault("FLAGS_enable_pir_api", "0")

    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(str(img_path))
    if not result:
        return ""

    lines: List[str] = []

    if isinstance(result, list):
        first_item = result[0] if result else None

        if isinstance(first_item, list):
            for line in first_item:
                if isinstance(line, (list, tuple)) and len(line) > 1:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and text_info:
                        lines.append(str(text_info[0]))

        elif isinstance(first_item, dict):
            rec_texts = first_item.get("rec_texts")
            if isinstance(rec_texts, list):
                lines = [str(t) for t in rec_texts]

    return "\n".join(lines)


def run_trocr(img_path: Path, model_name: str) -> str:
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def _safe_run(fn: Callable[[], str]) -> tuple[str, str]:
    try:
        return fn(), "ok"
    except Exception as exc:
        return f"", f"error: {exc}"


def _write_text(file_path: Path, text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


def _write_summary(results: List[OCRResult], summary_md: Path, summary_csv: Path) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    engines = sorted({r.engine for r in results})
    image_names = sorted({r.image_name for r in results})

    lines: List[str] = [
        "# OCR Benchmark Summary",
        "",
        "## Execution Results",
        "",
        "| Engine | Image | Status | Time (s) | Output File | Message |",
        "|---|---|---|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row.engine} | {row.image_name} | {row.status} | {row.duration_sec:.2f} | {row.output_file} | {row.message} |"
        )

    lines.extend(
        [
            "",
            "## Manual Scoring Table (Fill after review)",
            "",
            "| Engine | "
            + " | ".join(f"{img} Accuracy (1-10)" for img in image_names)
            + " | Avg | Notes |",
            "|---|" + "---|" * (len(image_names) + 3),
        ]
    )
    for engine in engines:
        placeholders = " | ".join([" "] * len(image_names))
        lines.append(f"| {engine} | {placeholders} |  |  |")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Winning engine: `TBD after manual scoring`",
            "- Mandatory fallback in pipeline: `Tesseract`",
        ]
    )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["engine", "image", "status", "duration_sec", "output_file", "message"])
        for row in results:
            writer.writerow(
                [
                    row.engine,
                    row.image_name,
                    row.status,
                    f"{row.duration_sec:.4f}",
                    row.output_file,
                    row.message,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR benchmark on sample images.")
    parser.add_argument(
        "--images-dir",
        default="data/sample_images",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark/results",
        help="Directory to write OCR outputs and summaries",
    )
    parser.add_argument(
        "--trocr-model",
        default="microsoft/trocr-base-handwritten",
        help="TrOCR model identifier",
    )
    parser.add_argument(
        "--disable-trocr",
        action="store_true",
        help="Skip TrOCR (useful when running fully offline)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    images_dir = (project_root / args.images_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = _load_images(images_dir)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    engines: Dict[str, Callable[[Path], str]] = {
        "tesseract_default": lambda p: run_tesseract(p, config=""),
        "tesseract_lstm_psm6": lambda p: run_tesseract(p, config="--oem 1 --psm 6"),
        "easyocr": run_easyocr,
        "paddleocr": run_paddleocr,
    }
    if not args.disable_trocr:
        engines["trocr"] = lambda p: run_trocr(p, model_name=args.trocr_model)

    all_results: List[OCRResult] = []

    print(f"[INFO] Found {len(images)} images in {images_dir}")
    print(f"[INFO] Results directory: {results_dir}")

    for image_idx, image_path in enumerate(images, start=1):
        for engine_name, engine_fn in engines.items():
            output_filename = f"{engine_name}_image{image_idx}.txt"
            output_path = results_dir / output_filename
            started = time.perf_counter()

            print(f"[RUN] {engine_name} on {image_path.name}")
            text, status_msg = _safe_run(lambda fn=engine_fn, p=image_path: fn(p))
            duration = time.perf_counter() - started

            if status_msg == "ok":
                _write_text(output_path, text)
                status = "success"
                message = ""
            else:
                _write_text(output_path, "")
                status = "failed"
                message = status_msg

            all_results.append(
                OCRResult(
                    engine=engine_name,
                    image_name=image_path.name,
                    output_file=str(output_path.relative_to(project_root)),
                    duration_sec=duration,
                    status=status,
                    message=message,
                )
            )
            print(f"[DONE] {engine_name} on {image_path.name} -> {status} ({duration:.2f}s)")

    summary_md = results_dir / "summary.md"
    summary_csv = results_dir / "summary.csv"
    _write_summary(all_results, summary_md, summary_csv)
    print(f"[INFO] Summary written to {summary_md}")
    print(f"[INFO] CSV written to {summary_csv}")

    failed_count = sum(1 for r in all_results if r.status != "success")
    if failed_count:
        print(f"[WARN] {failed_count} benchmark runs failed. Check summary.md for details.")
        return 2

    print("[OK] Benchmark completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

