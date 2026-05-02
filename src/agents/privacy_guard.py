"""
Privacy Guard — PII Detection & Data Protection
=================================================
Before processing an image through OCR, the Privacy Guard scans for potential
personally identifiable information (PII) visible in the image and warns the
user. It also enforces data-handling policies.

This module implements:
  - Post-OCR text scanning for PII patterns (emails, phone numbers, IDs)
  - Warning generation for the user
  - Optional redaction of PII in the final output
  - File cleanup guidance (don't leave sensitive data on disk)

Phase 2 Requirement Mapping:
  - Slide 30 (Ethical Agent Design: Privacy, User control)
  - Slide 15 (Legal Aspects: Data protection, User rights)
  - Slide 16 (IPR: Data protection awareness, GDPR mindset)
  - Slide 17 (Computer Crimes: Data theft risks)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PIIDetection:
    """A single PII detection result."""
    pii_type: str = ""       # "email", "phone", "cnic", "credit_card", etc.
    value: str = ""          # the detected value
    position: int = 0        # character position in text
    risk_level: str = "low"  # "low", "medium", "high"


@dataclass
class PrivacyReport:
    """Full privacy analysis report."""
    has_pii: bool = False
    detections: List[PIIDetection] = field(default_factory=list)
    risk_level: str = "none"   # "none", "low", "medium", "high"
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PrivacyGuard:
    """
    Scans OCR output text for PII and generates privacy warnings.

    Phase 1 had ZERO privacy awareness — any text extracted from images
    (including phone numbers, emails, ID numbers) was silently processed
    and saved with no warning. The agentic version proactively protects
    user data.
    """

    # Compiled regex patterns for PII detection
    PII_PATTERNS = {
        "email": {
            "pattern": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "risk": "medium",
            "label": "Email address",
        },
        "phone_intl": {
            "pattern": re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"),
            "risk": "medium",
            "label": "Phone number",
        },
        "cnic": {
            # Pakistani CNIC: 5 digits - 7 digits - 1 digit
            "pattern": re.compile(r"\d{5}-\d{7}-\d{1}"),
            "risk": "high",
            "label": "CNIC number",
        },
        "credit_card": {
            "pattern": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
            "risk": "high",
            "label": "Credit card number",
        },
        "date_of_birth": {
            "pattern": re.compile(r"\b(?:DOB|Date of Birth|Born)[\s:]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", re.IGNORECASE),
            "risk": "medium",
            "label": "Date of birth",
        },
        "passport": {
            "pattern": re.compile(r"\b[A-Z]{2}\d{7}\b"),
            "risk": "high",
            "label": "Passport number",
        },
    }

    def scan_text(self, text: str) -> PrivacyReport:
        """
        Scan extracted text for PII patterns.

        Returns a PrivacyReport with detections, risk level, and warnings.
        """
        report = PrivacyReport()

        if not text or not text.strip():
            return report

        for pii_type, config in self.PII_PATTERNS.items():
            matches = config["pattern"].finditer(text)
            for match in matches:
                detection = PIIDetection(
                    pii_type=pii_type,
                    value=self._mask_value(match.group(), config["risk"]),
                    position=match.start(),
                    risk_level=config["risk"],
                )
                report.detections.append(detection)

        if report.detections:
            report.has_pii = True

            # Determine overall risk level
            risk_levels = [d.risk_level for d in report.detections]
            if "high" in risk_levels:
                report.risk_level = "high"
            elif "medium" in risk_levels:
                report.risk_level = "medium"
            else:
                report.risk_level = "low"

            # Generate warnings
            report.warnings = self._generate_warnings(report)
            report.recommendations = self._generate_recommendations(report)

        return report

    def redact_text(self, text: str) -> str:
        """
        Replace all detected PII in text with redaction markers.
        Call this if the user opts in to automatic redaction.
        """
        result = text
        for pii_type, config in self.PII_PATTERNS.items():
            label = config["label"]
            result = config["pattern"].sub(f"[REDACTED {label.upper()}]", result)
        return result

    def _mask_value(self, value: str, risk: str) -> str:
        """Partially mask a PII value for display (show only last 4 chars)."""
        if risk == "high" and len(value) > 4:
            return "***" + value[-4:]
        elif len(value) > 6:
            return value[:3] + "***" + value[-3:]
        return "***"

    def _generate_warnings(self, report: PrivacyReport) -> List[str]:
        """Generate user-facing warnings."""
        warnings = []
        type_counts = {}
        for det in report.detections:
            type_counts[det.pii_type] = type_counts.get(det.pii_type, 0) + 1

        for pii_type, count in type_counts.items():
            label = self.PII_PATTERNS[pii_type]["label"]
            risk = self.PII_PATTERNS[pii_type]["risk"]
            warnings.append(
                f"⚠ {count} {label}(s) detected (risk: {risk}) — "
                f"review before saving"
            )

        if report.risk_level == "high":
            warnings.insert(
                0,
                "🔴 HIGH RISK: Sensitive personal data detected in this document. "
                "Do NOT share the output file without reviewing and redacting."
            )

        return warnings

    def _generate_recommendations(self, report: PrivacyReport) -> List[str]:
        """Generate actionable recommendations."""
        recs = [
            "Review the extracted text for any personal information",
            "Consider redacting PII before saving or sharing the document",
        ]

        if report.risk_level == "high":
            recs.append("Enable automatic PII redaction for this document")
            recs.append("Do not upload this document to cloud services without encryption")

        recs.append("Delete temporary processing files after conversion is complete")
        return recs
