"""
Orchestrator — Master Agent Controller
========================================
The Orchestrator is the top-level coordinator that implements the full
Observe → Interpret → Decide → Act → Learn cycle (Slide 25).

It wires together all agent components:
  - PerceptionAgent (Observe)
  - DecisionEngine (Decide)
  - Phase 1 Core (Act — OCR, formatting, docx generation)
  - FeedbackLoop (Learn)
  - MemoryStore (Remember)
  - AgentLogger (Log)
  - SafetyGuard (Validate)
  - PrivacyGuard (Protect)

Phase 2 Requirement Mapping:
  - Slide 25 (Operational Workflow: Observe → Interpret → Decide → Act → Learn)
  - Slide 23 (Agent Architecture: complete flow)
  - Slide 20 (Agentic System Concept)
  - Slide 22 (Agentic Vision: Tool → Agent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.agents.perception_agent import PerceptionAgent, ImageProfile
from src.agents.decision_engine import DecisionEngine, ProcessingStrategy
from src.agents.memory_store import MemoryStore, ConversionRecord
from src.agents.feedback_loop import FeedbackLoop, QualityReport
from src.agents.agent_logger import AgentLogger
from src.agents.safety_guard import SafetyGuard
from src.agents.privacy_guard import PrivacyGuard, PrivacyReport

from src.ocr_engine import OCREngine
from src.formatting_detector import FormattingDetector, FormattedBlock
from src.docx_generator import DocxGenerator
from src.preprocessing import (
    preprocess,
    preprocess_for_easyocr,
    preprocess_with_red_extraction,
)


@dataclass
class AgentResult:
    """Complete result of an agentic conversion."""
    text: str = ""
    blocks: list = field(default_factory=list)
    image_profile: Optional[ImageProfile] = None
    strategy: Optional[ProcessingStrategy] = None
    quality_report: Optional[QualityReport] = None
    privacy_report: Optional[PrivacyReport] = None
    corrections_applied: int = 0
    retry_count: int = 0
    success: bool = False
    error: str = ""


class Orchestrator:
    """
    Master agent that coordinates the full agentic pipeline.

    Phase 1 Pipeline (static):
        open image → preprocess → OCR → format → save

    Phase 2 Pipeline (agentic):
        OBSERVE (analyze image) →
        INTERPRET (perception profile) →
        DECIDE (strategy selection) →
        ACT (adaptive OCR + formatting) →
        EVALUATE (quality check) →
        RETRY? (if quality is poor) →
        PROTECT (privacy scan) →
        LEARN (store results, apply corrections) →
        OUTPUT

    The key difference: Phase 1 follows the SAME path every time.
    Phase 2 ADAPTS its path based on the image and past experience.
    """

    def __init__(
        self,
        memory: Optional[MemoryStore] = None,
        logger: Optional[AgentLogger] = None,
        safety: Optional[SafetyGuard] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        # Initialize all agent components
        self.memory = memory or MemoryStore()
        self.logger = logger or AgentLogger()
        self.safety = safety or SafetyGuard(autonomy_level="semi")

        self.perception = PerceptionAgent()
        self.decision = DecisionEngine(self.memory, self.logger, self.safety)
        self.feedback = FeedbackLoop(self.memory, self.logger)
        self.privacy = PrivacyGuard()

        self.formatter = FormattingDetector()
        self.doc_generator = DocxGenerator()

        # UI callbacks
        self._on_status = on_status or (lambda s: None)
        self._on_progress = on_progress or (lambda p: None)

        self.logger.log_decision(
            agent="Orchestrator",
            action="Agentic system initialized",
            reasoning="All agent modules loaded and ready",
            confidence=1.0,
        )

    def process_image(self, image_path: Path | str) -> AgentResult:
        """
        Full agentic pipeline: Observe → Decide → Act → Evaluate → Learn.

        This replaces the static Phase 1 _ocr_task() method.
        """
        result = AgentResult()
        path = Path(image_path)

        try:
            # ============================================================ #
            #  STEP 1: OBSERVE — Perception Agent analyzes the image        #
            # ============================================================ #
            self._on_status("🔍 Step 1/6: Analyzing image...")
            self._on_progress(10)

            profile = self.perception.analyze(path)
            result.image_profile = profile

            self.logger.log_decision(
                agent="PerceptionAgent",
                action=f"Image analyzed: quality={profile.quality_score}/100",
                reasoning=f"brightness={profile.brightness:.0f}, contrast={profile.contrast:.0f}, "
                          f"blur={profile.blur_score:.0f}, density={profile.density}",
                confidence=1.0,
                metadata={"recommendations": profile.recommendations},
            )

            # ============================================================ #
            #  STEP 2: DECIDE — Decision Engine selects strategy             #
            # ============================================================ #
            self._on_status("🧠 Step 2/6: Deciding optimal strategy...")
            self._on_progress(20)

            strategy = self.decision.decide_strategy(profile)
            result.strategy = strategy

            # ============================================================ #
            #  STEP 3: ACT — Execute OCR with the chosen strategy            #
            # ============================================================ #
            self._on_status(f"⚡ Step 3/6: Running {strategy.ocr_engine} OCR...")
            self._on_progress(40)

            text, boxes, confidence_values = self._execute_ocr(path, strategy)

            # ============================================================ #
            #  STEP 4: EVALUATE — Feedback Loop assesses quality             #
            # ============================================================ #
            self._on_status("📊 Step 4/6: Evaluating quality...")
            self._on_progress(60)

            quality = self.feedback.evaluate_quality(
                text, confidence_values, strategy.confidence_threshold
            )
            result.quality_report = quality

            # ---- Retry if quality is unacceptable ----
            attempt = 0
            while not quality.is_acceptable and attempt < strategy.max_retries:
                attempt += 1
                retry_strategy = self.decision.suggest_retry_strategy(
                    strategy, quality.overall_score, attempt
                )
                if retry_strategy is None:
                    break

                self._on_status(f"🔄 Retry {attempt}: Trying {retry_strategy.ocr_engine}...")

                retry_text, retry_boxes, retry_conf = self._execute_ocr(path, retry_strategy)
                retry_quality = self.feedback.evaluate_quality(
                    retry_text, retry_conf, retry_strategy.confidence_threshold
                )

                if retry_quality.overall_score > quality.overall_score:
                    text = retry_text
                    boxes = retry_boxes
                    confidence_values = retry_conf
                    quality = retry_quality
                    strategy = retry_strategy
                    result.retry_count = attempt

                    self.logger.log_decision(
                        agent="Orchestrator",
                        action=f"Retry {attempt} improved quality: {quality.overall_score:.2f}",
                        reasoning="Retry produced better results, adopting new output",
                        confidence=quality.overall_score,
                    )

            result.quality_report = quality
            result.strategy = strategy

            # ============================================================ #
            #  STEP 5: LEARN — Apply corrections & update memory             #
            # ============================================================ #
            self._on_status("📝 Step 5/6: Applying learned corrections...")
            self._on_progress(75)

            # Apply previously learned corrections
            text, corrections_applied = self.feedback.apply_learned_corrections(text)
            result.corrections_applied = corrections_applied

            # ============================================================ #
            #  STEP 6: PROTECT — Privacy scan                               #
            # ============================================================ #
            self._on_status("🔒 Step 6/6: Scanning for privacy concerns...")
            self._on_progress(85)

            privacy_report = self.privacy.scan_text(text)
            result.privacy_report = privacy_report

            if privacy_report.has_pii:
                self.logger.log_decision(
                    agent="PrivacyGuard",
                    action=f"PII detected: {len(privacy_report.detections)} item(s), risk={privacy_report.risk_level}",
                    reasoning="Sensitive data found in OCR output",
                    confidence=1.0,
                    metadata={"warnings": privacy_report.warnings},
                )

            # ============================================================ #
            #  FINALIZE — Build formatted blocks and store in memory         #
            # ============================================================ #
            self._on_status("✅ Finalizing...")
            self._on_progress(95)

            if boxes:
                result.blocks = self.formatter.detect_formatting(boxes)
            else:
                result.blocks = [
                    FormattedBlock(
                        text=text,
                        block_type="body",
                        alignment="left",
                        indent_level=0,
                    )
                ]

            result.text = text
            result.success = True

            # Store conversion in memory
            record = ConversionRecord(
                image_path=str(path),
                engine_used=strategy.ocr_engine,
                preprocessing_strategy=strategy.preprocessing,
                quality_score=quality.overall_score,
                confidence=quality.avg_confidence,
                image_brightness=profile.brightness,
                image_blur_score=profile.blur_score,
                success=True,
            )
            self.memory.add_conversion(record)

            self._on_progress(100)
            self._on_status("✅ Done — Agent pipeline complete")

        except Exception as e:
            result.success = False
            result.error = str(e)
            self._on_status(f"❌ Error: {e}")

            self.logger.log_decision(
                agent="Orchestrator",
                action="Pipeline failed",
                reasoning=str(e),
                confidence=0.0,
                outcome="error",
            )

        return result

    def save_document(
        self,
        blocks: list,
        output_path: Path | str,
        original_text: str = "",
        edited_text: str = "",
    ) -> bool:
        """
        Save the document — with safety check and learning.

        If the user edited the text before saving, the Feedback Loop
        learns from the corrections.
        """
        # Safety check for save action
        has_pii = False
        if edited_text:
            pii_report = self.privacy.scan_text(edited_text)
            has_pii = pii_report.has_pii

        check = self.safety.validate_action("save_document", {"has_pii": has_pii})

        # Learn from user corrections
        if original_text and edited_text and original_text != edited_text:
            corrections = self.feedback.learn_from_corrections(original_text, edited_text)

            # Update the last conversion record
            history = self.memory.get_history(limit=1)
            if history:
                last = history[0]
                last.was_corrected = True
                last.user_corrections = corrections

        # Generate the document
        success = self.doc_generator.generate_docx(blocks, output_path)

        self.logger.log_decision(
            agent="Orchestrator",
            action=f"Document saved to {output_path}",
            reasoning="User triggered save",
            confidence=1.0,
            outcome="success" if success else "failed",
        )

        return success

    def get_decision_log(self, limit: int = 20) -> str:
        """Return the human-readable decision log."""
        return self.logger.get_display_log(limit)

    def get_engine_stats(self) -> Dict[str, Any]:
        """Return engine performance statistics from memory."""
        return self.memory.get_engine_stats()

    # ================================================================ #
    #  Private: OCR execution with adaptive preprocessing                #
    # ================================================================ #

    def _execute_ocr(
        self,
        image_path: Path,
        strategy: ProcessingStrategy,
    ) -> Tuple[str, list, List[float]]:
        """
        Execute OCR using the strategy determined by the Decision Engine.

        This replaces the static Phase 1 approach of always using the
        same preprocessing + engine combination.
        """
        # ---- Adaptive preprocessing ----
        img = self._apply_adaptive_preprocessing(image_path, strategy)

        # ---- Run the selected OCR engine ----
        engine = OCREngine(engine=strategy.ocr_engine)
        text = engine.run(img)

        # Try to get boxes for formatting detection
        boxes = []
        confidence_values = []
        try:
            boxes = engine.run_with_boxes(img)
            confidence_values = [
                float(b.get("confidence", 0.0)) for b in boxes
            ]
        except (NotImplementedError, Exception):
            pass

        # ---- Dual-channel if enabled ----
        if strategy.use_dual_channel:
            try:
                _, red_img = preprocess_with_red_extraction(image_path)
                red_text = engine.run(red_img)
                if red_text.strip():
                    text = self._merge_dual_channel(text, red_text)

                    self.logger.log_decision(
                        agent="Orchestrator",
                        action="Dual-channel merge completed",
                        reasoning="Red ink text merged with main text",
                        confidence=0.8,
                    )
            except Exception:
                pass  # non-critical — fall back to single-channel

        return text, boxes, confidence_values

    def _apply_adaptive_preprocessing(
        self,
        image_path: Path,
        strategy: ProcessingStrategy,
    ) -> np.ndarray:
        """
        Apply preprocessing steps based on the strategy.
        
        Unlike Phase 1 (one-size-fits-all), this adapts based on the
        image profile and the Decision Engine's choices.
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if strategy.preprocessing == "minimal":
            # Minimal: just return the raw image (good quality images)
            self.logger.log_decision(
                agent="Orchestrator",
                action="Minimal preprocessing applied",
                reasoning="Image quality is good — minimal processing preserves detail",
                confidence=0.9,
            )
            return img

        # ---- Apply selected enhancements ----

        if strategy.apply_upscale:
            h, w = img.shape[:2]
            if w < 2000:
                scale = 2000 / float(w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if strategy.apply_brightness_enhance or strategy.apply_contrast_enhance:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if strategy.apply_contrast_enhance:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            if strategy.apply_brightness_enhance:
                # Adaptive brightness correction
                mean_brightness = float(np.mean(gray))
                if mean_brightness < 90:
                    gamma = 90.0 / max(1.0, mean_brightness)
                    gamma = min(gamma, 2.5)
                    table = np.array([
                        ((i / 255.0) ** (1.0 / gamma)) * 255
                        for i in range(256)
                    ]).astype("uint8")
                    gray = cv2.LUT(gray, table)

            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if strategy.apply_sharpening:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
            img = cv2.filter2D(img, -1, kernel)
            img = np.clip(img, 0, 255).astype(np.uint8)

        if strategy.apply_deskew:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 12,
            )
            coords = cv2.findNonZero(binary)
            if coords is not None and len(coords) > 100:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                if angle < -45:
                    angle = 90 + angle
                if abs(angle) > 0.2:
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img = cv2.warpAffine(
                        img, matrix, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

        return img

    def _merge_dual_channel(self, main_text: str, red_text: str) -> str:
        """Merge main and red-ink channel text, deduplicating."""
        merged_lines = []
        seen = set()
        for line in (main_text + "\n" + red_text).splitlines():
            clean = line.strip()
            key = clean.lower()
            if not clean or key in seen:
                continue
            seen.add(key)
            merged_lines.append(clean)
        return "\n".join(merged_lines)
