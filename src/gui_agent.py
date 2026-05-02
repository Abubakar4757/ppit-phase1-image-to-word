"""
Agentic GUI — Phase 2 Interface
=================================
A redesigned GUI that exposes the agentic system's capabilities:
  - Agent decision log panel (transparency/explainability)
  - Image quality profile display
  - Privacy warnings
  - Autonomy level controls
  - Engine performance stats
  - Human-in-the-loop override controls

This replaces the Phase 1 GUI (src/gui.py) which had no agent awareness.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from pathlib import Path
import threading
import sys

from PIL import Image, ImageTk

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.agents.orchestrator import Orchestrator, AgentResult
from src.agents.memory_store import MemoryStore
from src.agents.agent_logger import AgentLogger
from src.agents.safety_guard import SafetyGuard
from src.agents.perception_agent import ImageProfile
from src.agents.feedback_loop import QualityReport
from src.agents.privacy_guard import PrivacyReport
from src.formatting_detector import FormattedBlock
from src.preprocessing import preprocess


class AgenticApp:
    """
    Phase 2 GUI — Agent-Aware Interface.

    Differences from Phase 1 GUI:
      1. Shows the agent's decision log in real-time
      2. Displays image quality profile with recommendations
      3. Shows privacy warnings before saving
      4. Lets users control autonomy level
      5. Shows OCR engine performance statistics
      6. Supports human-in-the-loop confirmation for risky actions
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image-to-Word Converter — Agentic System (Phase 2)")
        self.root.geometry("1280x800")
        self.root.minsize(1000, 600)

        # State
        self.current_image_path = None
        self.last_result: AgentResult | None = None
        self.original_ocr_text = ""
        self.photo_original = None
        self.photo_preprocessed = None
        self._is_closing = False

        # Agent components
        self.memory = MemoryStore()
        self.logger = AgentLogger()
        self.safety = SafetyGuard(autonomy_level="semi")
        self.orchestrator = Orchestrator(
            memory=self.memory,
            logger=self.logger,
            safety=self.safety,
            on_status=lambda s: self._ui(lambda: self.status_var.set(s)),
            on_progress=lambda p: self._ui(lambda: self.progress.config(value=p)),
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the complete UI layout."""

        # ---- Menu Bar ----
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self._open_image)
        file_menu.add_command(label="Save .docx", command=self._save_docx)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        agent_menu = tk.Menu(menubar, tearoff=0)
        agent_menu.add_command(label="View Decision Log", command=self._show_decision_log)
        agent_menu.add_command(label="View Engine Stats", command=self._show_engine_stats)
        agent_menu.add_command(label="Clear Memory", command=self._clear_memory)
        menubar.add_cascade(label="Agent", menu=agent_menu)

        # ---- Top Toolbar ----
        toolbar = tk.Frame(self.root, pady=5, padx=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = tk.Button(toolbar, text="📂 Open Image", command=self._open_image, width=14)
        self.btn_open.pack(side=tk.LEFT, padx=3)

        self.btn_run = tk.Button(toolbar, text="🤖 Run Agent", command=self._run_agent_threaded, state=tk.DISABLED, width=14)
        self.btn_run.pack(side=tk.LEFT, padx=3)

        self.btn_save = tk.Button(toolbar, text="💾 Save .docx", command=self._save_docx, state=tk.DISABLED, width=14)
        self.btn_save.pack(side=tk.LEFT, padx=3)

        self.btn_clear = tk.Button(toolbar, text="🗑 Clear", command=self._clear, width=10)
        self.btn_clear.pack(side=tk.LEFT, padx=3)

        # Autonomy level selector
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        tk.Label(toolbar, text="Autonomy:").pack(side=tk.LEFT, padx=(5, 2))
        self.autonomy_var = tk.StringVar(value="semi")
        autonomy_combo = ttk.Combobox(
            toolbar, textvariable=self.autonomy_var,
            values=["full", "semi", "manual"], width=8, state="readonly",
        )
        autonomy_combo.pack(side=tk.LEFT)
        autonomy_combo.bind("<<ComboboxSelected>>", self._on_autonomy_change)

        # ---- Main Content Area ----
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === LEFT PANEL: Image Previews + Agent Info ===
        left_frame = tk.Frame(main_paned, width=350)
        main_paned.add(left_frame, minsize=300)

        # Image preview notebook (tabs)
        img_notebook = ttk.Notebook(left_frame)
        img_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Original image
        original_tab = tk.Frame(img_notebook)
        img_notebook.add(original_tab, text="Original")
        self.img_original_label = tk.Label(original_tab, text="No Image Selected", bg="#2d2d2d", fg="white")
        self.img_original_label.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Preprocessed image
        preproc_tab = tk.Frame(img_notebook)
        img_notebook.add(preproc_tab, text="Preprocessed")
        self.img_preprocessed_label = tk.Label(preproc_tab, text="No Preview", bg="#2d2d2d", fg="white")
        self.img_preprocessed_label.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Image Profile (agent perception data)
        profile_tab = tk.Frame(img_notebook)
        img_notebook.add(profile_tab, text="📊 Image Profile")
        self.profile_text = scrolledtext.ScrolledText(profile_tab, wrap=tk.WORD, font=("Consolas", 9))
        self.profile_text.pack(fill=tk.BOTH, expand=True)
        self.profile_text.insert(tk.END, "Run agent to see image analysis...")
        self.profile_text.config(state=tk.DISABLED)

        # === CENTER PANEL: Extracted Text ===
        center_frame = tk.Frame(main_paned)
        main_paned.add(center_frame, minsize=350)

        tk.Label(center_frame, text="Extracted Text (editable)", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.text_area = tk.Text(center_frame, wrap=tk.WORD, undo=True, font=("Consolas", 10))
        text_scroll = tk.Scrollbar(center_frame, command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=text_scroll.set)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # === RIGHT PANEL: Agent Log + Privacy Warnings ===
        right_frame = tk.Frame(main_paned, width=320)
        main_paned.add(right_frame, minsize=280)

        right_notebook = ttk.Notebook(right_frame)
        right_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Agent Decision Log
        log_tab = tk.Frame(right_notebook)
        right_notebook.add(log_tab, text="🧠 Agent Log")
        self.log_text = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, font=("Consolas", 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Agent decisions will appear here...\n")
        self.log_text.config(state=tk.DISABLED)

        # Tab 2: Privacy & Warnings
        privacy_tab = tk.Frame(right_notebook)
        right_notebook.add(privacy_tab, text="🔒 Privacy")
        self.privacy_text = scrolledtext.ScrolledText(privacy_tab, wrap=tk.WORD, font=("Consolas", 9))
        self.privacy_text.pack(fill=tk.BOTH, expand=True)
        self.privacy_text.insert(tk.END, "Privacy analysis will appear here...\n")
        self.privacy_text.config(state=tk.DISABLED)

        # Tab 3: Quality Report
        quality_tab = tk.Frame(right_notebook)
        right_notebook.add(quality_tab, text="📊 Quality")
        self.quality_text = scrolledtext.ScrolledText(quality_tab, wrap=tk.WORD, font=("Consolas", 9))
        self.quality_text.pack(fill=tk.BOTH, expand=True)
        self.quality_text.insert(tk.END, "Quality report will appear here...\n")
        self.quality_text.config(state=tk.DISABLED)

        # ---- Bottom Status Bar ----
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Ready — Open an image to begin")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, anchor=tk.W, font=("Segoe UI", 9))
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=250, mode="determinate")
        self.progress.pack(side=tk.RIGHT, padx=5, pady=2)

    # ================================================================ #
    #  Image loading                                                     #
    # ================================================================ #

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return

        self.current_image_path = Path(path)
        self.last_result = None
        self.original_ocr_text = ""

        self._display_original_preview(self.current_image_path)
        self.btn_run.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self.status_var.set(f"Loaded: {self.current_image_path.name}")
        self.text_area.delete("1.0", tk.END)
        self.progress["value"] = 0

    def _display_original_preview(self, path: Path) -> None:
        img = Image.open(path)
        img.thumbnail((400, 700))
        self.photo_original = ImageTk.PhotoImage(img)
        self.img_original_label.config(image=self.photo_original, text="")

    # ================================================================ #
    #  Agent execution                                                   #
    # ================================================================ #

    def _run_agent_threaded(self) -> None:
        self.btn_run.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_var.set("🤖 Agent starting...")

        thread = threading.Thread(target=self._agent_task, daemon=True)
        thread.start()

    def _agent_task(self) -> None:
        try:
            result = self.orchestrator.process_image(self.current_image_path)
            self.last_result = result
            self.original_ocr_text = result.text

            # Update all UI panels
            self._ui(lambda: self._update_text_area(result.text))
            self._ui(lambda: self._update_profile_panel(result.image_profile))
            self._ui(lambda: self._update_log_panel())
            self._ui(lambda: self._update_privacy_panel(result.privacy_report))
            self._ui(lambda: self._update_quality_panel(result.quality_report))

            if result.success:
                self._ui(lambda: self.btn_save.config(state=tk.NORMAL))
                self._ui(lambda: self.status_var.set("✅ Agent pipeline complete"))
            else:
                self._ui(lambda: messagebox.showerror("Error", f"Agent failed: {result.error}"))

        except Exception as e:
            self._ui(lambda: messagebox.showerror("Error", f"Agent error: {str(e)}"))
            self._ui(lambda: self.status_var.set("❌ Failed"))
        finally:
            self._ui(lambda: self.btn_run.config(state=tk.NORMAL))
            self._ui(lambda: self.btn_open.config(state=tk.NORMAL))

    # ================================================================ #
    #  UI update helpers                                                 #
    # ================================================================ #

    def _update_text_area(self, text: str) -> None:
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, text)

    def _update_profile_panel(self, profile: ImageProfile | None) -> None:
        self.profile_text.config(state=tk.NORMAL)
        self.profile_text.delete("1.0", tk.END)

        if not profile:
            self.profile_text.insert(tk.END, "No profile available")
            self.profile_text.config(state=tk.DISABLED)
            return

        lines = [
            "═══ Image Analysis Profile ═══",
            "",
            f"  Resolution:    {profile.width} × {profile.height}",
            f"  Brightness:    {profile.brightness:.1f} {'⚠ DARK' if profile.is_dark else '✓'}",
            f"  Contrast:      {profile.contrast:.1f} {'⚠ LOW' if profile.is_low_contrast else '✓'}",
            f"  Blur Score:    {profile.blur_score:.1f} {'⚠ BLURRY' if profile.is_blurry else '✓'}",
            f"  Skew Angle:    {profile.skew_angle:.1f}° {'⚠ SKEWED' if profile.is_skewed else '✓'}",
            f"  Text Density:  {profile.density}",
            f"  Color Profile: {profile.dominant_color}",
            "",
            f"  ★ Quality Score: {profile.quality_score}/100",
            "",
            "─── Recommendations ───",
        ]
        for rec in profile.recommendations:
            lines.append(f"  • {rec}")

        self.profile_text.insert(tk.END, "\n".join(lines))
        self.profile_text.config(state=tk.DISABLED)

    def _update_log_panel(self) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, self.logger.get_display_log(30))
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _update_privacy_panel(self, report: PrivacyReport | None) -> None:
        self.privacy_text.config(state=tk.NORMAL)
        self.privacy_text.delete("1.0", tk.END)

        if not report:
            self.privacy_text.insert(tk.END, "No privacy analysis yet.")
            self.privacy_text.config(state=tk.DISABLED)
            return

        if not report.has_pii:
            self.privacy_text.insert(tk.END, "✅ No PII detected in extracted text.\n\n")
            self.privacy_text.insert(tk.END, "The document appears safe to save and share.")
        else:
            lines = [
                f"⚠ PRIVACY ALERT — Risk Level: {report.risk_level.upper()}",
                "",
                "─── Detections ───",
            ]
            for det in report.detections:
                lines.append(f"  • {det.pii_type}: {det.value} (risk: {det.risk_level})")

            lines.append("")
            lines.append("─── Warnings ───")
            for warning in report.warnings:
                lines.append(f"  {warning}")

            lines.append("")
            lines.append("─── Recommendations ───")
            for rec in report.recommendations:
                lines.append(f"  • {rec}")

            self.privacy_text.insert(tk.END, "\n".join(lines))

        self.privacy_text.config(state=tk.DISABLED)

    def _update_quality_panel(self, report: QualityReport | None) -> None:
        self.quality_text.config(state=tk.NORMAL)
        self.quality_text.delete("1.0", tk.END)

        if not report:
            self.quality_text.insert(tk.END, "No quality report yet.")
            self.quality_text.config(state=tk.DISABLED)
            return

        status = "✅ PASS" if report.is_acceptable else "❌ FAIL"
        lines = [
            f"═══ OCR Quality Report ═══",
            "",
            f"  Overall Score:   {report.overall_score:.2f} {status}",
            f"  Avg Confidence:  {report.avg_confidence:.2f}",
            f"  Words Extracted: {report.word_count}",
            f"  Lines Extracted: {report.line_count}",
            f"  Gibberish Ratio: {report.gibberish_ratio:.1%}",
            f"  Short Word Ratio:{report.short_word_ratio:.1%}",
        ]

        if report.issues:
            lines.append("")
            lines.append("─── Issues ───")
            for issue in report.issues:
                lines.append(f"  ⚠ {issue}")

        lines.append("")
        lines.append(f"  💡 {report.suggestion}")

        if self.last_result and self.last_result.retry_count > 0:
            lines.append("")
            lines.append(f"  🔄 Retries used: {self.last_result.retry_count}")

        if self.last_result and self.last_result.api_ocr_used:
            lines.append("")
            lines.append("  🌐 Vision API escalation used")
            lines.append("     Local OCR quality was too low after retries.")
            lines.append("     Agent autonomously called gpt-4o-mini Vision API.")
            lines.append("     Image was sent to OpenAI — see Privacy tab for details.")

        if self.last_result and self.last_result.corrections_applied > 0:
            lines.append(f"  📝 Learned corrections applied: {self.last_result.corrections_applied}")

        self.quality_text.insert(tk.END, "\n".join(lines))
        self.quality_text.config(state=tk.DISABLED)

    # ================================================================ #
    #  Save                                                              #
    # ================================================================ #

    def _save_docx(self) -> None:
        if not self.last_result and not self.text_area.get("1.0", tk.END).strip():
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Word Document", "*.docx")],
        )
        if not path:
            return

        edited_text = self.text_area.get("1.0", tk.END).strip()

        # Build blocks from edited text
        blocks = [
            FormattedBlock(
                text=edited_text,
                block_type="body",
                alignment="left",
                indent_level=0,
            )
        ]

        success = self.orchestrator.save_document(
            blocks=blocks,
            output_path=path,
            original_text=self.original_ocr_text,
            edited_text=edited_text,
        )

        if success:
            messagebox.showinfo("Success", f"Document saved to {path}")
            # Refresh the log to show any learning that happened
            self._update_log_panel()
        else:
            messagebox.showerror("Error", "Failed to save document.")

    # ================================================================ #
    #  Agent menu actions                                                #
    # ================================================================ #

    def _show_decision_log(self) -> None:
        """Show full decision log in a new window."""
        log_window = tk.Toplevel(self.root)
        log_window.title("Agent Decision Log")
        log_window.geometry("700x500")

        text = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, font=("Consolas", 9))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, self.logger.get_display_log(100))
        text.config(state=tk.DISABLED)

    def _show_engine_stats(self) -> None:
        """Show engine performance statistics."""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("OCR Engine Performance Stats")
        stats_window.geometry("500x400")

        text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        stats = self.orchestrator.get_engine_stats()
        lines = ["═══ OCR Engine Performance ═══", ""]
        for engine, data in stats.items():
            lines.append(f"  {engine.upper()}")
            lines.append(f"    Total Runs:      {data['total_runs']}")
            lines.append(f"    Avg Confidence:  {data['avg_confidence']:.2f}")
            lines.append("")

        # Show best engine recommendation
        best = self.memory.get_best_engine()
        if best:
            lines.append(f"  ★ Recommended Engine: {best} (best historical confidence)")
        else:
            lines.append("  No enough data yet for a recommendation.")

        text.insert(tk.END, "\n".join(lines))
        text.config(state=tk.DISABLED)

    def _clear_memory(self) -> None:
        if messagebox.askyesno("Clear Memory", "This will reset all learned patterns and preferences. Continue?"):
            import os
            if self.memory.memory_path.exists():
                os.remove(self.memory.memory_path)
            self.memory = MemoryStore()
            self.orchestrator.memory = self.memory
            self.orchestrator.decision.memory = self.memory
            self.orchestrator.feedback.memory = self.memory
            messagebox.showinfo("Memory Cleared", "Agent memory has been reset.")

    def _on_autonomy_change(self, event=None) -> None:
        level = self.autonomy_var.get()
        self.safety.set_autonomy_level(level)
        self.logger.log_decision(
            agent="User",
            action=f"Autonomy level changed to '{level}'",
            reasoning="User preference",
            confidence=1.0,
        )
        self._update_log_panel()

    # ================================================================ #
    #  Clear & Utility                                                   #
    # ================================================================ #

    def _clear(self) -> None:
        self.current_image_path = None
        self.last_result = None
        self.original_ocr_text = ""
        self.photo_original = None
        self.photo_preprocessed = None
        self.img_original_label.config(image="", text="No Image Selected")
        self.img_preprocessed_label.config(image="", text="No Preview")
        self.text_area.delete("1.0", tk.END)
        self.btn_run.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.status_var.set("Ready — Open an image to begin")
        self.progress["value"] = 0

        # Clear log display
        self.logger.clear_session()
        self._update_log_panel()

        # Reset readonly text areas
        for widget in [self.profile_text, self.privacy_text, self.quality_text]:
            widget.config(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, "Cleared.")
            widget.config(state=tk.DISABLED)

    def _ui(self, callback) -> None:
        """Thread-safe UI update."""
        if self._is_closing:
            return
        try:
            if self.root.winfo_exists():
                self.root.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def _on_close(self) -> None:
        self._is_closing = True
        self.memory.save()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = AgenticApp(root)
    root.mainloop()
