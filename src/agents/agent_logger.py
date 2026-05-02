"""
Agent Logger — Decision Audit Trail
=====================================
Every autonomous decision the agent makes is logged with a timestamp,
the reasoning behind it, the alternatives considered, and the outcome.

This serves three Phase 2 requirements:
  1. **Transparency** (Slide 30 — Ethical Agent Design)
  2. **Explainability** (Slide 32 — Safety Mechanisms)
  3. **Override capability** (Slide 32 — users can review and reverse decisions)

Logs are written both to an in-memory buffer (for UI display) and to a
persistent log file on disk for post-run audit.
"""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default log file location
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "agent_decisions.jsonl"


@dataclass
class DecisionLogEntry:
    """A single logged decision."""
    timestamp: str = ""
    agent: str = ""             # which agent made the decision
    action: str = ""            # what action was taken
    reasoning: str = ""         # why this action was chosen
    alternatives: list = field(default_factory=list)  # what other options existed
    confidence: float = 0.0     # how confident the agent was
    outcome: str = ""           # result of the action
    metadata: dict = field(default_factory=dict)  # extra context

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_display_string(self) -> str:
        """Human-readable single-line summary."""
        return (
            f"[{self.timestamp}] {self.agent}: {self.action} "
            f"(reason: {self.reasoning}) → {self.outcome}"
        )


class AgentLogger:
    """
    Centralized logging for all agent decisions.

    Phase 1 had zero logging — decisions were invisible.
    The agentic system logs every autonomous decision so that:
      - Users can understand WHY the agent did something
      - Decisions can be audited after the fact
      - Incorrect decisions can be traced and corrected
    """

    def __init__(self, log_path: Optional[Path | str] = None) -> None:
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self._entries: List[DecisionLogEntry] = []

    def log_decision(
        self,
        agent: str,
        action: str,
        reasoning: str,
        alternatives: Optional[List[str]] = None,
        confidence: float = 1.0,
        outcome: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DecisionLogEntry:
        """
        Log an autonomous decision.

        Parameters
        ----------
        agent : str
            Name of the agent component (e.g., "PerceptionAgent", "DecisionEngine")
        action : str
            What action was taken (e.g., "Selected EasyOCR engine")
        reasoning : str
            Why this action was chosen (e.g., "Image has good contrast, EasyOCR performs best")
        alternatives : list
            What other options were available
        confidence : float
            Agent's self-assessed confidence (0.0 – 1.0)
        outcome : str
            Result (filled in after action completes)
        metadata : dict
            Any additional context
        """
        entry = DecisionLogEntry(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            action=action,
            reasoning=reasoning,
            alternatives=alternatives or [],
            confidence=confidence,
            outcome=outcome,
            metadata=metadata or {},
        )

        self._entries.append(entry)
        self._persist_entry(entry)
        return entry

    def update_outcome(self, entry: DecisionLogEntry, outcome: str) -> None:
        """Update the outcome of a previously logged decision."""
        entry.outcome = outcome
        # Re-persist the updated entry
        self._persist_entry(entry)

    def get_entries(self, limit: int = 50) -> List[DecisionLogEntry]:
        """Return recent log entries (in-memory)."""
        return self._entries[-limit:]

    def get_display_log(self, limit: int = 20) -> str:
        """Return a human-readable log string for UI display."""
        entries = self._entries[-limit:]
        if not entries:
            return "No decisions logged yet."

        lines = ["═══ Agent Decision Log ═══", ""]
        for entry in entries:
            lines.append(f"▸ [{entry.timestamp[11:19]}] {entry.agent}")
            lines.append(f"  Action: {entry.action}")
            lines.append(f"  Reason: {entry.reasoning}")
            if entry.alternatives:
                lines.append(f"  Alternatives: {', '.join(entry.alternatives)}")
            lines.append(f"  Confidence: {entry.confidence:.0%}")
            if entry.outcome:
                lines.append(f"  Outcome: {entry.outcome}")
            lines.append("")

        return "\n".join(lines)

    def clear_session(self) -> None:
        """Clear in-memory entries (log file is preserved)."""
        self._entries.clear()

    def _persist_entry(self, entry: DecisionLogEntry) -> None:
        """Append a single entry to the JSONL log file."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except IOError:
            pass  # non-critical — don't crash if logging fails
