"""Объединение слоёв детекции и разрешение пересечений.

Несколько детекторов могут найти пересекающиеся сущности (например, regex
поймал телефон, а NER случайно зацепил часть как локацию). Разрешаем конфликт
детерминированно: структурный слой важнее NER, длиннее важнее короче.
"""

from __future__ import annotations

from .base import Detector, Entity, EntityType

# Приоритет типов при пересечении: чем больше — тем важнее. Структурированные
# идентификаторы с контрольной суммой почти всегда вернее, чем догадка NER.
_PRIORITY: dict[EntityType, int] = {
    EntityType.SNILS: 100,
    EntityType.INN: 100,
    EntityType.OGRN: 100,
    EntityType.CARD: 95,
    EntityType.ACCOUNT: 95,
    EntityType.PASSPORT: 90,
    EntityType.EMAIL: 90,
    EntityType.PHONE: 90,
    EntityType.CAR_PLATE: 80,
    EntityType.IP: 70,
    EntityType.DOB: 70,
    EntityType.PERSON: 50,
    EntityType.ORG: 40,
    EntityType.LOCATION: 30,
}


def _rank(e: Entity) -> tuple:
    """Ключ сортировки «важности» сущности при конфликте (больше = победитель)."""
    return (_PRIORITY.get(e.type, 0), e.confidence, e.length)


class DetectionPipeline:
    """Прогоняет текст через все детекторы и отдаёт непересекающийся список."""

    def __init__(self, detectors: list[Detector]) -> None:
        self._detectors = detectors

    def detect(self, text: str) -> list[Entity]:
        raw: list[Entity] = []
        for d in self._detectors:
            raw.extend(d.detect(text))
        return self._resolve(raw)

    def _resolve(self, entities: list[Entity]) -> list[Entity]:
        # Жадно: сортируем по важности, берём сущность, отбрасываем всё, что с ней
        # пересекается. Так структурные ПД вытесняют слабые догадки NER.
        ordered = sorted(entities, key=_rank, reverse=True)
        kept: list[Entity] = []
        for cand in ordered:
            if any(cand.overlaps(k) for k in kept):
                continue
            kept.append(cand)
        kept.sort(key=lambda e: e.start)
        return kept
