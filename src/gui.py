import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
from PIL import Image, ImageTk
import sys

# Add src to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.ocr_engine import OCREngine
from src.formatting_detector import FormattedBlock, FormattingDetector
from src.docx_generator import DocxGenerator
from src.preprocessing import preprocess

class ImageToWordApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Notes to Word Converter")
        self.root.geometry("1000x700")
        
        # State
        self.current_image_path = None
        self.ocr_engine = OCREngine(engine="auto")
        self.detector = FormattingDetector()
        self.generator = DocxGenerator()
        self.last_blocks = []
        self.photo_original = None
        self.photo_preprocessed = None
        self._is_closing = False

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self._setup_ui()

    def _setup_ui(self):
        # 1. Top Toolbar
        toolbar = tk.Frame(self.root, pady=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_open = tk.Button(toolbar, text="Open Image", command=self._open_image)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        self.btn_run = tk.Button(toolbar, text="Run OCR", command=self._run_ocr_threaded, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        
        self.btn_save = tk.Button(toolbar, text="Save .docx", command=self._save_docx, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear = tk.Button(toolbar, text="Clear", command=self._clear)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        tk.Label(toolbar, text="Engine:").pack(side=tk.LEFT, padx=(20, 5))
        self.engine_var = tk.StringVar(value="auto")
        self.engine_combo = ttk.Combobox(toolbar, textvariable=self.engine_var, values=["auto", "easyocr", "tesseract"], width=10)
        self.engine_combo.pack(side=tk.LEFT)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)
        
        # 2. Main Content (Paned Window)
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Original Image Preview
        original_frame = tk.Frame(paned)
        paned.add(original_frame)
        tk.Label(original_frame, text="Original").pack(side=tk.TOP, anchor=tk.W)
        self.img_original_label = tk.Label(original_frame, text="No Image Selected", bg="gray80", width=35)
        self.img_original_label.pack(fill=tk.BOTH, expand=True)

        # Middle: Preprocessed Image Preview
        pre_frame = tk.Frame(paned)
        paned.add(pre_frame)
        tk.Label(pre_frame, text="Preprocessed").pack(side=tk.TOP, anchor=tk.W)
        self.img_preprocessed_label = tk.Label(pre_frame, text="No Preprocessed Preview", bg="gray85", width=35)
        self.img_preprocessed_label.pack(fill=tk.BOTH, expand=True)

        # Right: Extracted Text
        text_frame = tk.Frame(paned)
        paned.add(text_frame)
        
        self.text_area = tk.Text(text_frame, wrap=tk.WORD, undo=True)
        scrollbar = tk.Scrollbar(text_frame, command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=scrollbar.set)
        
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 3. Bottom Status Bar
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=5, pady=2)

    def _open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
            
        self.current_image_path = Path(path)
        self.last_blocks = []
        self._display_original_preview(self.current_image_path)
        self._display_preprocessed_preview(self.current_image_path)
        self.btn_run.config(state=tk.NORMAL)
        self.status_var.set(f"Loaded: {self.current_image_path.name}")
        self.text_area.delete("1.0", tk.END)
        self.btn_save.config(state=tk.DISABLED)

    def _display_original_preview(self, path):
        img = Image.open(path)
        img.thumbnail((300, 600))
        self.photo_original = ImageTk.PhotoImage(img)
        self.img_original_label.config(image=self.photo_original, text="")

    def _display_preprocessed_preview(self, path):
        processed = preprocess(path)
        if len(processed.shape) == 2:
            preview = Image.fromarray(processed)
        else:
            rgb = processed[:, :, ::-1]
            preview = Image.fromarray(rgb)
        preview.thumbnail((300, 600))
        self.photo_preprocessed = ImageTk.PhotoImage(preview)
        self.img_preprocessed_label.config(image=self.photo_preprocessed, text="")

    def _on_engine_change(self, event=None):
        self.ocr_engine = OCREngine(engine=self.engine_var.get())

    def _run_ocr_threaded(self):
        self.btn_run.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.status_var.set("Processing...")
        
        thread = threading.Thread(target=self._ocr_task, daemon=True)
        thread.start()

    def _ui(self, callback):
        if self._is_closing:
            return
        try:
            if self.root.winfo_exists():
                self.root.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def _on_close(self):
        self._is_closing = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _ocr_task(self):
        try:
            # 1. Loading
            self._ui(lambda: self.status_var.set("Step 1/3: Loading image..."))
            self._ui(lambda: self.progress.config(value=10))
            raw_image_path = self.current_image_path
            self._ui(lambda: self.progress.config(value=20))
            
            # 2. OCR — run on RAW image (preprocessing hurts EasyOCR on handwriting)
            self._ui(lambda: self.status_var.set("Step 2/3: Running OCR (this may take a moment)..."))
            # Use run() for well-ordered, cleaned text
            cleaned_text = self.ocr_engine.run(raw_image_path)
            # Use run_with_boxes() for layout detection
            boxes = self.ocr_engine.run_with_boxes(raw_image_path)
            self._ui(lambda: self.progress.config(value=80))
            
            # 3. Formatting detection from boxes (for docx structure)
            self._ui(lambda: self.status_var.set("Step 3/3: Detecting layout..."))
            self.last_blocks = self.detector.detect_formatting(boxes)
            self._ui(lambda: self.progress.config(value=100))
            
            # Show the CLEAN text in UI (not the raw box concatenation)
            self._ui(lambda: self._update_text_area(cleaned_text))
            self._ui(lambda: self.status_var.set("Done"))
            self._ui(lambda: self.btn_save.config(state=tk.NORMAL))
            
        except Exception as e:
            self._ui(lambda: messagebox.showerror("Error", f"OCR failed: {str(e)}"))
            self._ui(lambda: self.status_var.set("Failed"))
        finally:
            self._ui(lambda: self.btn_run.config(state=tk.NORMAL))
            self._ui(lambda: self.btn_open.config(state=tk.NORMAL))

    def _update_text_area(self, text):
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, text)

    def _save_docx(self):
        if not self.last_blocks and not self.text_area.get("1.0", tk.END).strip():
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".docx", filetypes=[("Word Document", "*.docx")])
        if not path:
            return
            
        edited_text = self.text_area.get("1.0", tk.END).strip()
        blocks_to_save = self.last_blocks

        if edited_text:
            blocks_to_save = [
                FormattedBlock(
                    text=edited_text,
                    block_type="body",
                    alignment="left",
                    indent_level=0,
                )
            ]

        success = self.generator.generate_docx(blocks_to_save, path)
        if success:
            messagebox.showinfo("Success", f"Saved to {path}")
        else:
            messagebox.showerror("Error", "Failed to save file.")

    def _clear(self):
        self.current_image_path = None
        self.last_blocks = []
        self.photo_original = None
        self.photo_preprocessed = None
        self.img_original_label.config(image="", text="No Image Selected")
        self.img_preprocessed_label.config(image="", text="No Preprocessed Preview")
        self.text_area.delete("1.0", tk.END)
        self.btn_run.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.status_var.set("Ready")
        self.progress['value'] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToWordApp(root)
    root.mainloop()
