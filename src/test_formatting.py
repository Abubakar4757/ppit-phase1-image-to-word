import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.ocr_engine import OCREngine
from src.formatting_detector import FormattingDetector

def test_formatting_detection():
    print("Testing Formatting Detection...")
    engine = OCREngine(engine="auto")
    detector = FormattingDetector()
    
    # Path to a sample image
    sample_image = project_root / "data" / "sample_images" / "image1.jpeg"
    if not sample_image.exists():
        sample_image = project_root / "data" / "sample_images" / "image1.jpg"
        
    if not sample_image.exists():
        print(f"Error: Sample image not found at {sample_image}")
        return False

    print(f"Processing image: {sample_image.name}")
    
    # Get OCR boxes
    boxes = engine.run_with_boxes(sample_image)
    
    if not boxes:
        print("Error: OCR extracted no boxes.")
        return False
        
    print(f"Extracted {len(boxes)} boxes. Detecting formatting...")
    
    blocks = detector.detect_formatting(boxes)
    
    if not blocks:
        print("Error: No formatted blocks detected.")
        return False
        
    print(f"Detected {len(blocks)} blocks.")
    print("-" * 30)
    for i, block in enumerate(blocks[:10]):
        display_text = block.text[:50].replace('\n', ' ')
        print(f"Block {i+1}: [{block.block_type}] {display_text}...")
        print(f"  Alignment: {block.alignment}, Indent: {block.indent_level}, Y: {block.line_y:.1f}")
    print("-" * 30)
    
    return True

if __name__ == "__main__":
    success = test_formatting_detection()
    sys.exit(0 if success else 1)
