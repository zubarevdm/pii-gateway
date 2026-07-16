"""Vault — локальная карта маппинга «плейсхолдер ↔ оригинальное значение».

КРИТИЧНО ДЛЯ 152-ФЗ: эта карта — единственное место, где реальные ПД и их
плейсхолдеры лежат вместе. Она НИКОГДА не уходит в облако. Живёт в памяти
процесса в контуре клиента и существует ровно столько, сколько нужно для
восстановления ответа. По умолчанию не персистится на диск.

Один Vault соответствует одному диалогу (conversation): благодаря этому одно и
то же лицо получает один и тот же плейсхолдер во всех сообщениях, и модель
сохраняет связность («Иванов» всегда [PERSON_1]).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.detectors.base import EntityType


@dataclass
class Vault:
    """Двунаправленная карта плейсхолдеров для одного диалога."""

    # value -> placeholder (чтобы одинаковые ПД давали один плейсхолдер)
    _by_value: dict[str, str] = field(default_factory=dict)
    # placeholder -> value (для восстановления ответа модели)
    _by_placeholder: dict[str, str] = field(default_factory=dict)
    # счётчики по типам: PERSON -> 2 означает, что выдано [PERSON_1], [PERSON_2]
    _counters: dict[EntityType, int] = field(default_factory=dict)

    def placeholder_for(self, etype: EntityType, value: str) -> str:
        """Вернуть стабильный плейсхолдер для значения, создав при первой встрече.

        Нормализуем по типу+значению: «+7 900...» и «+7900...» считаем одним
        номером, чтобы не плодить плейсхолдеры. Имена нормализуем по регистру.
        """
        key = self._normalize(etype, value)
        existing = self._by_value.get(key)
        if existing is not None:
            return existing

        idx = self._counters.get(etype, 0) + 1
        self._counters[etype] = idx
        placeholder = f"[{etype.value}_{idx}]"

        self._by_value[key] = placeholder
        self._by_placeholder[placeholder] = value
        return placeholder

    def original(self, placeholder: str) -> str | None:
        return self._by_placeholder.get(placeholder)

    def placeholders(self) -> list[str]:
        return list(self._by_placeholder.keys())

    def __len__(self) -> int:
        return len(self._by_placeholder)

    @staticmethod
    def _normalize(etype: EntityType, value: str) -> str:
        if etype in (EntityType.PHONE, EntityType.CARD, EntityType.ACCOUNT,
                     EntityType.INN, EntityType.SNILS, EntityType.OGRN):
            digits = "".join(c for c in value if c.isdigit())
            return f"{etype.value}:{digits}"
        return f"{etype.value}:{value.casefold().strip()}"
