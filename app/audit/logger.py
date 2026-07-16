"""Аудит обезличивания — доказательная база для проверки РКН.

ПРИНЦИП: лог фиксирует ФАКТ и ТИП маскировки, но НИКОГДА сами персональные
данные. Запись «в 14:03 в запрос req-123 к gpt-4o было обезличено 2 ФИО и
1 телефон» доказывает, что фильтр сработал, и при этом сам лог не является
хранилищем ПД (иначе мы бы создали новую базу ПД — ровно то, чего избегаем).

Формат — JSON Lines: одна запись на строку, удобно грепать и показывать
проверяющему.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from app.detectors.base import Entity


@dataclass
class AuditRecord:
    ts: str                       # ISO-время события
    request_id: str
    model: str                    # какая модель запрашивалась
    text_chars: int               # длина исходного текста (без содержимого!)
    masked_total: int             # сколько всего сущностей замаскировано
    masked_by_type: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class AuditLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled

    def record(
        self,
        request_id: str,
        model: str,
        text_chars: int,
        entities: list[Entity],
    ) -> AuditRecord:
        by_type = Counter(e.type.value for e in entities)
        rec = AuditRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            request_id=request_id,
            model=model,
            text_chars=text_chars,
            masked_total=len(entities),
            masked_by_type=dict(by_type),
        )
        if self._enabled:
            self._append(rec)
        return rec

    def _append(self, rec: AuditRecord) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")
