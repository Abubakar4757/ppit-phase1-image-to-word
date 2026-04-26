import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.ocr_engine import OCREngine

def test_ocr_extraction():
    print("Testing OCREngine extraction...")
    engine = OCREngine(engine="auto")
    
    # Path to a sample image
    sample_image = project_root / "data" / "sample_images" / "image1.jpeg"
    if not sample_image.exists():
        # Try different extension
        sample_image = project_root / "data" / "sample_images" / "image1.jpg"
        
    if not sample_image.exists():
        print(f"Error: Sample image not found at {sample_image}")
        return False

    print(f"Processing image: {sample_image.name}")
    
    text = engine.run(sample_image)
    
    if text and len(text.strip()) > 0:
        print("Success: OCR extracted text.")
        print("-" * 20)
        print(text[:200] + "..." if len(text) > 200 else text)
        print("-" * 20)
        
        # Check for subscript reconstruction
        if any(c in text for c in "₀₁₂₃₄₅₆₇₈₉"):
            print("Success: Subscripts reconstructed.")
        else:
            print("Note: No subscripts detected in this sample.")
            
        return True
    else:
        print("Error: OCR extracted no text.")
        return False

if __name__ == "__main__":
    success = test_ocr_extraction()
    sys.exit(0 if success else 1)
