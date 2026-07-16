"""Тесты разрешения пересечений в pipeline."""

from app.detectors.base import Entity, EntityType
from app.detectors.pipeline import DetectionPipeline


class _Fixed:
    """Детектор, возвращающий заранее заданные сущности."""

    def __init__(self, name: str, ents: list[Entity]) -> None:
        self.name = name
        self._ents = ents

    def detect(self, text: str) -> list[Entity]:
        return list(self._ents)


def test_structured_beats_ner_on_overlap():
    # regex поймал телефон 0..15, NER ошибочно зацепил 5..10 как локацию.
    phone = Entity(0, 15, EntityType.PHONE, "+7 900 123 4567", 1.0, "regex")
    loc = Entity(5, 10, EntityType.LOCATION, "900 1", 0.85, "natasha")
    pipe = DetectionPipeline([_Fixed("a", [loc]), _Fixed("b", [phone])])
    out = pipe.detect("x" * 20)
    assert len(out) == 1
    assert out[0].type == EntityType.PHONE


def test_non_overlapping_kept():
    a = Entity(0, 5, EntityType.PERSON, "Иван", 0.85, "natasha")
    b = Entity(10, 20, EntityType.PHONE, "+79001234567", 1.0, "regex")
    pipe = DetectionPipeline([_Fixed("x", [a, b])])
    out = pipe.detect("y" * 25)
    assert len(out) == 2
    # отсортированы по началу
    assert out[0].start < out[1].start
