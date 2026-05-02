"""
Safety Guard — Decision Validation & Override
===============================================
The Safety Guard sits between the Decision Engine and the Action Executor.
It validates every autonomous decision before execution and provides
mechanisms for human override.

Key responsibilities:
  - Validate that agent decisions are within safe bounds
  - Prevent over-automation (e.g., auto-saving without user consent)
  - Provide a human-in-the-loop checkpoint for high-risk decisions
  - Allow users to override any agent decision

Phase 2 Requirement Mapping:
  - Slide 29 (Human-in-the-Loop)
  - Slide 31 (Risk Assessment)
  - Slide 32 (Safety Mechanisms: Logging, Override, Explainability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


@dataclass
class SafetyCheck:
    """Result of a safety validation."""
    is_safe: bool = True
    risk_level: str = "none"       # "none", "low", "medium", "high"
    requires_human_approval: bool = False
    warnings: List[str] = field(default_factory=list)
    blocked_reason: str = ""


class SafetyGuard:
    """
    Validates agent decisions and enforces safety boundaries.

    Phase 1 had no safety layer — whatever the code did, it did silently.
    The agentic version explicitly checks every decision against safety
    rules before executing it.

    Autonomy Levels (Slide 28):
      - FULL: Agent acts without asking (low-risk actions only)
      - SEMI: Agent proposes, user confirms (medium-risk)
      - MANUAL: User must explicitly initiate (high-risk)
    """

    # Actions classified by risk level
    RISK_CLASSIFICATIONS = {
        # Low risk — agent can act autonomously
        "select_ocr_engine": "low",
        "adjust_preprocessing": "low",
        "enhance_brightness": "low",
        "enhance_contrast": "low",
        "apply_sharpening": "low",
        "deskew_image": "low",
        "upscale_image": "low",
        "retry_with_different_engine": "low",

        # Medium risk — agent proposes, user confirms
        "save_document": "medium",
        "apply_corrections": "medium",
        "use_api_ocr": "medium",            # sends data externally
        "enable_dual_channel": "medium",

        # High risk — requires explicit user action
        "delete_files": "high",
        "share_document": "high",
        "send_to_cloud": "high",
        "auto_redact_pii": "high",
    }

    def __init__(self, autonomy_level: str = "semi") -> None:
        """
        Parameters
        ----------
        autonomy_level : str
            "full", "semi", or "manual"
        """
        self.autonomy_level = autonomy_level
        self._overrides: Dict[str, str] = {}  # action -> user decision ("allow"/"block")

    def validate_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> SafetyCheck:
        """
        Validate whether an action is safe to execute autonomously.

        Returns a SafetyCheck indicating whether the action is approved,
        needs human approval, or is blocked.
        """
        check = SafetyCheck()
        context = context or {}

        # Check if user has explicitly overridden this action
        if action in self._overrides:
            override = self._overrides[action]
            if override == "block":
                check.is_safe = False
                check.blocked_reason = f"Action '{action}' blocked by user override"
                return check
            elif override == "allow":
                check.is_safe = True
                return check

        # Get risk level
        risk = self.RISK_CLASSIFICATIONS.get(action, "medium")
        check.risk_level = risk

        # Apply autonomy level rules
        if self.autonomy_level == "manual":
            # Everything needs human approval
            check.requires_human_approval = True
            check.warnings.append(f"Manual mode: '{action}' requires your approval")

        elif self.autonomy_level == "semi":
            if risk in ("medium", "high"):
                check.requires_human_approval = True
                check.warnings.append(
                    f"Action '{action}' (risk: {risk}) requires your confirmation"
                )

        elif self.autonomy_level == "full":
            if risk == "high":
                check.requires_human_approval = True
                check.warnings.append(
                    f"High-risk action '{action}' still requires confirmation even in full-auto mode"
                )

        # Additional context-specific checks
        if action == "use_api_ocr":
            check.warnings.append(
                "This action sends your image to an external API (OpenAI). "
                "Ensure the image contains no sensitive data."
            )

        if action == "save_document" and context.get("has_pii", False):
            check.warnings.append(
                "Document contains detected PII. Review before saving."
            )
            check.requires_human_approval = True

        return check

    def set_override(self, action: str, decision: str) -> None:
        """
        Allow the user to permanently override an action's safety classification.

        Parameters
        ----------
        action : str
            The action to override
        decision : str
            "allow" or "block"
        """
        if decision in ("allow", "block"):
            self._overrides[action] = decision

    def clear_override(self, action: str) -> None:
        """Remove a user override."""
        self._overrides.pop(action, None)

    def get_autonomy_level(self) -> str:
        """Return current autonomy level."""
        return self.autonomy_level

    def set_autonomy_level(self, level: str) -> None:
        """Change autonomy level."""
        if level in ("full", "semi", "manual"):
            self.autonomy_level = level

    def get_risk_summary(self) -> str:
        """Return a human-readable summary of risk classifications."""
        lines = ["═══ Action Risk Classifications ═══", ""]
        for risk_level in ("low", "medium", "high"):
            actions = [a for a, r in self.RISK_CLASSIFICATIONS.items() if r == risk_level]
            emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}[risk_level]
            lines.append(f"{emoji} {risk_level.upper()} RISK:")
            for action in actions:
                override = self._overrides.get(action, "—")
                lines.append(f"   • {action} (override: {override})")
            lines.append("")

        return "\n".join(lines)
