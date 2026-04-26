import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.ocr_engine import OCREngine
from src.formatting_detector import FormattingDetector
from src.docx_generator import DocxGenerator

def test_full_pipeline():
    print("Testing Full Pipeline (OCR -> Formatting -> DOCX)...")
    engine = OCREngine(engine="auto")
    detector = FormattingDetector()
    generator = DocxGenerator()
    
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = project_root / "data" / "sample_images"
    images = list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg"))
    
    if not images:
        print(f"Error: No images found in {images_dir}")
        return False

    for image_path in images:
        print(f"\nProcessing image: {image_path.name}")
        
        # 1. OCR
        text = engine.run(image_path)
        boxes = engine.run_with_boxes(image_path)
        if not boxes:
            print(f"  Warning: No text found in {image_path.name}")
            continue
        if not text.strip():
            print(f"  Warning: No text content reconstructed in {image_path.name}")
            
        # 2. Detect Formatting
        blocks = detector.detect_formatting(boxes)
        print(f"  Detected {len(blocks)} blocks.")
        
        # 3. Generate DOCX
        output_name = image_path.stem + ".docx"
        output_path = output_dir / output_name
        
        success = generator.generate_docx(blocks, output_path)
        if success:
            print(f"  Success: Saved to {output_path}")
        else:
            print(f"  Error: Failed to save {output_path}")
            
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
