"""
Image-to-Word Converter — Phase 2: Agentic System
===================================================
Entry point for the Phase 2 agentic application.

Phase 1 launched a static GUI with manual OCR pipeline.
Phase 2 launches an agent-aware GUI where the system autonomously:
  - Analyzes image quality (Perception Agent)
  - Selects optimal OCR strategy (Decision Engine)
  - Evaluates output quality and retries if needed (Feedback Loop)
  - Learns from user corrections (Memory Store)
  - Scans for privacy concerns (Privacy Guard)
  - Logs all decisions for transparency (Agent Logger)
  - Enforces safety boundaries (Safety Guard)

Usage:
  python main.py            → Launch Phase 2 agentic GUI
  python main.py --phase1   → Launch original Phase 1 GUI
"""

import sys
import tkinter as tk


def main():
    # Allow fallback to Phase 1 if requested
    if "--phase1" in sys.argv:
        from src.gui import ImageToWordApp
        root = tk.Tk()
        app = ImageToWordApp(root)
        root.mainloop()
        return

    # Phase 2: Agentic System
    from src.gui_agent import AgenticApp
    root = tk.Tk()
    app = AgenticApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
