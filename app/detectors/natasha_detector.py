"""Слой 2: контекстный RU NER на Natasha (ФИО, организации, локации).

В свободном тексте regex бессилен: «позвони Иванову» или «директор завода
в Туле» — это ПД, но без формата. Natasha (slovnet NER + эмбеддинги navec)
извлекает PER/ORG/LOC контекстно, полностью локально, без обращения в сеть.

Модель тяжёлая на загрузку (~1–2 c, кэшируется навсегда после первого вызова),
поэтому грузим лениво и держим один экземпляр на процесс.
"""

from __future__ import annotations

import threading

from .base import Detector, Entity, EntityType

# Маппинг тегов Natasha -> наши категории.
_TAG_MAP = {
    "PER": EntityType.PERSON,
    "ORG": EntityType.ORG,
    "LOC": EntityType.LOCATION,
}


class _Engine:
    """Ленивая обёртка над тяжёлыми объектами Natasha, потокобезопасная."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            from natasha import (
                Doc,
                NewsEmbedding,
                NewsNERTagger,
                Segmenter,
            )

            self._Doc = Doc
            self._segmenter = Segmenter()
            emb = NewsEmbedding()
            self._ner = NewsNERTagger(emb)
            self._ready = True

    def spans(self, text: str) -> list[tuple[int, int, str]]:
        self._ensure()
        doc = self._Doc(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._ner)
        return [(s.start, s.stop, s.type) for s in doc.spans]


_engine = _Engine()


class NatashaNERDetector:
    """Контекстный детектор ФИО/организаций/локаций. Реализует Detector."""

    name = "natasha"

    def __init__(self, confidence: float = 0.85) -> None:
        # NER не даёт калиброванной вероятности — используем фиксированную
        # уверенность ниже regex, чтобы при пересечении побеждал структурный слой.
        self.confidence = confidence

    def detect(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        for start, stop, tag in _engine.spans(text):
            etype = _TAG_MAP.get(tag)
            if etype is None:
                continue
            found.append(
                Entity(
                    start=start,
                    end=stop,
                    type=etype,
                    value=text[start:stop],
                    confidence=self.confidence,
                    source=self.name,
                )
            )
        return found

    def warmup(self) -> None:
        """Прогрев модели на старте, чтобы первый реальный запрос не тормозил."""
        _engine.spans("Прогрев модели Иван Иванов Москва.")


# Статический типчек.
_: Detector = NatashaNERDetector()
