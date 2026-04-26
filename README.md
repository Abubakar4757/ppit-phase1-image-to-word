# PPIT Phase 1 Image-to-Word Converter

Convert single-page handwritten note images (`.jpg`, `.jpeg`, `.png`) into editable Word documents (`.docx`) using OCR, layout detection, and document generation.

## Requirements
- Python `3.10+`
- Tesseract OCR binary (UB-Mannheim build for Windows)
- Project dependencies from `requirements.txt`

## Installation
1. Create virtual environment:
   - `python -m venv venv`
2. Activate virtual environment (PowerShell):
   - `venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `pip install -r requirements.txt`

## Tesseract Path Setup (Windows)
Install Tesseract from UB-Mannheim and ensure this path exists:
- `C:\Program Files\Tesseract-OCR\tesseract.exe`

The project auto-detects Tesseract at that location. If installed elsewhere, add it to `PATH` or update `pytesseract.pytesseract.tesseract_cmd` accordingly.

## Run the Application
From project root:
- `python main.py`

## Run Benchmark
From project root:
- `cd benchmark && python run_benchmark.py`

## Known Limitations
- Handwriting OCR accuracy depends on image quality and writing style.
- Very dense notes, stylized handwriting, and uncommon symbols may still produce recognition errors.
- TrOCR support is limited for full-page notes and remains secondary to EasyOCR in this phase.
