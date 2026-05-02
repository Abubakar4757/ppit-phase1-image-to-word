"""
Memory Store — Persistent Agent Memory
========================================
Implements both short-term (session) and long-term (disk-persisted) memory
so the agent can learn from past conversions and user preferences.

Short-term memory tracks the current session's decisions and results.
Long-term memory persists across sessions as a JSON file, storing:
  - User preferences (preferred engine, preprocessing settings)
  - Conversion history with quality scores
  - Correction patterns (user-edited text vs OCR output)

Phase 2 Requirement Mapping:
  - Slide 27 (Memory & Context: Short-term vs Long-term)
  - Slide 25 (Operational Workflow: Learn)
  - Slide 23 (Agent Architecture: Memory)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default path for persistent memory file
DEFAULT_MEMORY_PATH = Path(__file__).resolve().parents[2] / "data" / "agent_memory.json"


@dataclass
class ConversionRecord:
    """A single conversion event stored in long-term memory."""
    timestamp: str = ""
    image_path: str = ""
    engine_used: str = ""
    preprocessing_strategy: str = ""
    quality_score: float = 0.0
    confidence: float = 0.0
    was_corrected: bool = False
    user_corrections: int = 0
    image_brightness: float = 0.0
    image_blur_score: float = 0.0
    success: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "ConversionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MemoryStore:
    """
    Dual-layer memory system: session (volatile) + persistent (JSON).

    The Phase 1 app had zero memory — every run started from scratch.
    The agentic version remembers what worked, what failed, and what the
    user prefers, allowing the Decision Engine to improve over time.
    """

    def __init__(self, memory_path: Optional[Path | str] = None) -> None:
        self.memory_path = Path(memory_path) if memory_path else DEFAULT_MEMORY_PATH

        # ---- Short-term (session) memory ----
        self.session_decisions: List[Dict[str, Any]] = []
        self.session_start = datetime.now().isoformat()

        # ---- Long-term (persistent) memory ----
        self._data: Dict[str, Any] = {
            "preferences": {
                "preferred_engine": None,       # user's last-chosen engine
                "auto_enhance": True,           # whether agent auto-enhances images
                "confidence_threshold": 0.5,    # minimum acceptable confidence
            },
            "history": [],                      # list of ConversionRecord dicts
            "correction_patterns": {},          # {misspelled: corrected} learned pairs
            "engine_performance": {             # aggregated stats per engine
                "easyocr": {"total_runs": 0, "avg_confidence": 0.0, "total_confidence": 0.0},
                "tesseract": {"total_runs": 0, "avg_confidence": 0.0, "total_confidence": 0.0},
                "paddleocr": {"total_runs": 0, "avg_confidence": 0.0, "total_confidence": 0.0},
                "trocr": {"total_runs": 0, "avg_confidence": 0.0, "total_confidence": 0.0},
            },
        }

        self._load()

    # ================================================================ #
    #  Persistence                                                       #
    # ================================================================ #

    def _load(self) -> None:
        """Load long-term memory from disk."""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Merge loaded data with defaults (in case new keys were added)
                for key in self._data:
                    if key in loaded:
                        if isinstance(self._data[key], dict) and isinstance(loaded[key], dict):
                            self._data[key].update(loaded[key])
                        else:
                            self._data[key] = loaded[key]
            except (json.JSONDecodeError, IOError):
                pass  # corrupted file — start fresh

    def save(self) -> None:
        """Persist long-term memory to disk."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ================================================================ #
    #  Short-term memory (session)                                       #
    # ================================================================ #

    def record_session_decision(self, decision: Dict[str, Any]) -> None:
        """Store a decision made during this session."""
        decision["session_timestamp"] = datetime.now().isoformat()
        self.session_decisions.append(decision)

    def get_session_decisions(self) -> List[Dict[str, Any]]:
        """Return all decisions from the current session."""
        return self.session_decisions.copy()

    # ================================================================ #
    #  Long-term memory — conversion history                              #
    # ================================================================ #

    def add_conversion(self, record: ConversionRecord) -> None:
        """Add a conversion record to history and update engine stats."""
        record.timestamp = datetime.now().isoformat()
        self._data["history"].append(record.to_dict())

        # Update engine performance stats
        engine = record.engine_used.lower()
        if engine in self._data["engine_performance"]:
            stats = self._data["engine_performance"][engine]
            stats["total_runs"] += 1
            stats["total_confidence"] += record.confidence
            stats["avg_confidence"] = (
                stats["total_confidence"] / stats["total_runs"]
            )

        # Keep history bounded (last 200 conversions)
        if len(self._data["history"]) > 200:
            self._data["history"] = self._data["history"][-200:]

        self.save()

    def get_history(self, limit: int = 20) -> List[ConversionRecord]:
        """Return recent conversion history."""
        raw = self._data["history"][-limit:]
        return [ConversionRecord.from_dict(r) for r in raw]

    # ================================================================ #
    #  Long-term memory — preferences                                     #
    # ================================================================ #

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self._data["preferences"].get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference and persist."""
        self._data["preferences"][key] = value
        self.save()

    # ================================================================ #
    #  Long-term memory — correction patterns (learning)                   #
    # ================================================================ #

    def learn_correction(self, wrong: str, correct: str) -> None:
        """Store a user correction so the agent can apply it in future runs."""
        self._data["correction_patterns"][wrong.lower()] = correct
        self.save()

    def get_corrections(self) -> Dict[str, str]:
        """Return all learned correction patterns."""
        return dict(self._data["correction_patterns"])

    # ================================================================ #
    #  Long-term memory — engine performance                               #
    # ================================================================ #

    def get_best_engine(self) -> Optional[str]:
        """
        Return the engine with the highest average confidence,
        or None if no data exists.
        """
        best_engine = None
        best_avg = 0.0

        for engine, stats in self._data["engine_performance"].items():
            if stats["total_runs"] > 0 and stats["avg_confidence"] > best_avg:
                best_avg = stats["avg_confidence"]
                best_engine = engine

        return best_engine

    def get_engine_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return performance stats for all engines."""
        return dict(self._data["engine_performance"])
