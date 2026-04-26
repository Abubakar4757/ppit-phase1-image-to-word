
# OCR Benchmark (Phase 1)

## Goal
Benchmark free OCR engines on the 3 provided WhatsApp handwritten chemistry images and select the best engine for the main pipeline.

## Engines Targeted
- Tesseract (default)
- Tesseract (`--oem 1 --psm 6`)
- EasyOCR
- PaddleOCR
- TrOCR (`microsoft/trocr-base-handwritten` by default)

## Run
From project root:
- `python benchmark/run_benchmark.py`

Results generated in:
- `benchmark/results/*.txt`
- `benchmark/results/summary.md`
- `benchmark/results/summary.csv`

## Manual Scoring Rubric (1-10 each)
- Accuracy: correctness of recognized words
- Structure: line breaks/section order readability
- Chemical Terms: handling of chemistry vocabulary/symbols

Select winner based on highest practical quality and acceptable runtime.

## Actual Run Summary (This Machine)

### Environment
- Python: `3.10.0`
- Tesseract binary: `5.5.0.20241111`
- Hardware mode: CPU only

### Engine Execution Status
- Tesseract (default): ✅ success on all 3 images
- Tesseract (`--oem 1 --psm 6`): ✅ success on all 3 images
- EasyOCR: ✅ success on all 3 images
- TrOCR (`microsoft/trocr-base-handwritten`): ✅ success on all 3 images
- PaddleOCR: ❌ failed on all 3 images due Paddle runtime error:
  - `ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`

### Quick Quality Notes (manual visual review)
- EasyOCR produced the most usable full-page outputs among successful engines.
- Tesseract (both configs) produced longer text but with heavier noise on handwriting.
- TrOCR output was too short/incomplete for these full notebook-page images without tiling.

## Phase 1 Decision
- **Winning engine for Phase 3 integration:** `EasyOCR`
- **Mandatory fallback engine:** `Tesseract` (project requirement)

## Follow-up for PaddleOCR (optional)
- Retry in a clean env with alternate `paddlepaddle` build/version.
- If still failing, keep PaddleOCR out of default pipeline and document as compatibility limitation.

## Quality Improvement Pass

### Run
From project root:
- `python benchmark/run_quality_pass.py`

Results are generated in:
- `benchmark/quality_results/*.txt`
- `benchmark/quality_results/summary.csv`
- `benchmark/quality_results/summary.md`

### Modes Compared
- `tesseract_raw`
- `tesseract_preprocessed`
- `easyocr_raw`
- `easyocr_cleaned`
- `easyocr_preprocessed`

### Current Best Practical Output
- `easyocr_cleaned` is currently the best readability/quality trade-off in this project state.
- `easyocr_preprocessed` underperforms on these notebook images (heavy degradation), so it is not selected.

### Remaining Quality Limitations
- Handwriting ambiguity remains high for chemistry symbols and small cursive text.
- Two-column notebooks still need stronger layout reconstruction for perfect reading order.
- Final production pipeline should combine OCR with line-grouping + chemistry-aware correction rules.
