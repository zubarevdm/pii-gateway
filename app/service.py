"""Сборка ядра: детекторы -> pipeline -> anonymizer (+ аудит).

Один экземпляр на процесс. Тяжёлая модель Natasha грузится один раз.
"""

from __future__ import annotations

from functools import lru_cache

from app.anonymizer import Anonymizer
from app.audit import AuditLogger
from app.config import Settings, get_settings
from app.detectors.base import Detector
from app.detectors.pipeline import DetectionPipeline
from app.detectors.regex_detector import RegexDetector


def _build_pipeline(settings: Settings) -> DetectionPipeline:
    detectors: list[Detector] = [RegexDetector()]
    if settings.enable_ner:
        # Импортируем здесь, чтобы тесты regex не тянули тяжёлую модель.
        from app.detectors.natasha_detector import NatashaNERDetector

        detectors.append(NatashaNERDetector(confidence=settings.ner_confidence))
    return DetectionPipeline(detectors)


class GatewayService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.pipeline = _build_pipeline(self.settings)
        self.anonymizer = Anonymizer(self.pipeline)
        self.audit = AuditLogger(self.settings.audit_path, self.settings.audit_enabled)

    def warmup(self) -> None:
        if self.settings.enable_ner:
            from app.detectors.natasha_detector import NatashaNERDetector

            for d in self.pipeline._detectors:  # noqa: SLF001 — внутренний прогрев
                if isinstance(d, NatashaNERDetector):
                    d.warmup()


@lru_cache(maxsize=1)
def get_service() -> GatewayService:
    return GatewayService()
