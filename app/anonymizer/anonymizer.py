"""Anonymizer — связывает детекцию, токенизацию и восстановление.

anonymize(): текст с ПД -> обезличенный текст + заполненный Vault.
restore():   ответ модели с плейсхолдерами -> текст с восстановленными ПД.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.detectors.base import Entity
from app.detectors.pipeline import DetectionPipeline

from .vault import Vault

# Плейсхолдер вида [PERSON_1]. Восстановление терпимо к возможным искажениям
# регистра/пробелов, которые иногда вносит модель.
_PLACEHOLDER_RE = re.compile(r"\[\s*([A-Z_]+_\d+)\s*\]")


@dataclass
class AnonymizeResult:
    """Результат обезличивания одного текста."""

    text: str               # обезличенный текст (уходит в облако)
    entities: list[Entity]  # что нашли (для аудита; значения остаются локально)
    replaced: int           # сколько подстановок сделано


class Anonymizer:
    def __init__(self, pipeline: DetectionPipeline) -> None:
        self._pipeline = pipeline

    def anonymize(self, text: str, vault: Vault) -> AnonymizeResult:
        """Заменить все найденные ПД на плейсхолдеры, наполнив vault.

        Идём справа налево, чтобы не сдвигать индексы ещё не обработанных
        сущностей при замене.
        """
        entities = self._pipeline.detect(text)
        if not entities:
            return AnonymizeResult(text=text, entities=[], replaced=0)

        # Сначала в порядке появления присваиваем плейсхолдеры, чтобы нумерация
        # шла естественно ([PERSON_1] — первое встреченное лицо).
        forward = sorted(entities, key=lambda e: e.start)
        placeholders = {
            id(e): vault.placeholder_for(e.type, e.value) for e in forward
        }

        # Затем заменяем справа налево, чтобы не сдвигать индексы.
        out = text
        for ent in sorted(entities, key=lambda e: e.start, reverse=True):
            out = out[: ent.start] + placeholders[id(ent)] + out[ent.end:]

        return AnonymizeResult(text=out, entities=entities, replaced=len(entities))

    def restore(self, text: str, vault: Vault) -> str:
        """Заменить плейсхолдеры обратно на оригинальные ПД по карте vault."""
        def _sub(m: re.Match) -> str:
            placeholder = f"[{m.group(1)}]"
            original = vault.original(placeholder)
            return original if original is not None else m.group(0)

        return _PLACEHOLDER_RE.sub(_sub, text)
