"""Слой 1: детекция структурированных ПД по regex + контрольным суммам.

Эти данные имеют формат и ловятся правилами почти со 100% точностью. Где есть
контрольная сумма (ИНН, СНИЛС, ОГРН, карта) — проверяем её, чтобы не маскировать
случайные числа. Это базовая ценность, которая работает мгновенно и без сети.
"""

from __future__ import annotations

import re

from .base import Detector, Entity, EntityType
from .validators import valid_inn, valid_luhn, valid_ogrn, valid_snils

# Телефоны РФ: +7/8, опц. скобки/дефисы/пробелы. Требуем разделители или префикс,
# чтобы не хватать случайные 11-значные числа.
_PHONE = re.compile(
    r"(?<!\d)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Паспорт РФ: 4 цифры серия + 6 цифр номер. Часто с ключевым словом рядом —
# но ловим и «голый» формат «12 34 567890».
_PASSPORT = re.compile(
    r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)"
)
_PASSPORT_KW = re.compile(
    r"(?i)(?:паспорт|серия)[^\d]{0,15}(\d{2}\s?\d{2})\s*(?:номер|№|N)?\s*(\d{6})"
)

# СНИЛС: XXX-XXX-XXX YY
_SNILS = re.compile(r"(?<!\d)\d{3}[\-\s]\d{3}[\-\s]\d{3}[\s\-]\d{2}(?!\d)")

# ИНН: 10 или 12 цифр подряд (валидируем контрольной суммой).
_INN = re.compile(r"(?<!\d)\d{12}(?!\d)|(?<!\d)\d{10}(?!\d)")

# ОГРН/ОГРНИП: 13 или 15 цифр.
_OGRN = re.compile(r"(?<!\d)\d{15}(?!\d)|(?<!\d)\d{13}(?!\d)")

# Банковская карта: 16 (реже 13-19) цифр, группами по 4 или слитно.
_CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")

# Расчётный/лицевой счёт: 20 цифр.
_ACCOUNT = re.compile(r"(?<!\d)\d{20}(?!\d)")

# Госномер авто РФ: А123ВС 77 (буквы кириллической «латиницы» ABEKMHOPCTYX).
_CAR_PLATE = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])[АВЕКМНОРСТУХABEKMHOPCTYX]\d{3}"
    r"[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\s?\d{2,3}(?![A-Za-zА-Яа-я0-9])"
)

_IP = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
)

# Дата рождения: при ключевом слове рядом, либо явный формат ДД.ММ.ГГГГ.
_DATE = re.compile(r"(?<!\d)\d{2}[.\-/]\d{2}[.\-/]\d{4}(?!\d)")
_DOB_KW = re.compile(
    r"(?i)(?:д\.?\s?р\.?|дата\s+рожд\w*|род(?:ился|илась|\.)?|г\.?\s?р\.?)"
)


class RegexDetector:
    """Детектор структурированных ПД. Реализует протокол Detector."""

    name = "regex"

    def detect(self, text: str) -> list[Entity]:
        found: list[Entity] = []

        for m in _EMAIL.finditer(text):
            found.append(self._ent(m.start(), m.end(), EntityType.EMAIL, m.group()))

        for m in _PHONE.finditer(text):
            found.append(self._ent(m.start(), m.end(), EntityType.PHONE, m.group()))

        for m in _SNILS.finditer(text):
            if valid_snils(m.group()):
                found.append(self._ent(m.start(), m.end(), EntityType.SNILS, m.group()))

        # Паспорт с ключевым словом — приоритетнее «голого» формата.
        passport_spans: list[tuple[int, int]] = []
        for m in _PASSPORT_KW.finditer(text):
            s, e = m.start(1), m.end(2)
            found.append(self._ent(s, e, EntityType.PASSPORT, text[s:e]))
            passport_spans.append((s, e))

        for m in _IP.finditer(text):
            found.append(self._ent(m.start(), m.end(), EntityType.IP, m.group(), 0.9))

        for m in _CAR_PLATE.finditer(text):
            found.append(self._ent(m.start(), m.end(), EntityType.CAR_PLATE, m.group()))

        # Счёт (20) — до карт/ИНН, чтобы не дробить.
        for m in _ACCOUNT.finditer(text):
            found.append(self._ent(m.start(), m.end(), EntityType.ACCOUNT, m.group()))

        for m in _OGRN.finditer(text):
            if valid_ogrn(m.group()):
                found.append(self._ent(m.start(), m.end(), EntityType.OGRN, m.group()))

        for m in _CARD.finditer(text):
            if valid_luhn(m.group()):
                found.append(self._ent(m.start(), m.end(), EntityType.CARD, m.group()))

        for m in _INN.finditer(text):
            if valid_inn(m.group()):
                found.append(self._ent(m.start(), m.end(), EntityType.INN, m.group()))

        # «Голый» паспорт 4+6, если не перекрыт паспортом-с-ключевым-словом.
        for m in _PASSPORT.finditer(text):
            if not any(s <= m.start() < e for s, e in passport_spans):
                found.append(
                    self._ent(m.start(), m.end(), EntityType.PASSPORT, m.group(), 0.7)
                )

        # Даты: повышаем до DOB, если рядом ключевое слово рождения.
        for m in _DATE.finditer(text):
            window = text[max(0, m.start() - 25): m.start()]
            if _DOB_KW.search(window):
                found.append(self._ent(m.start(), m.end(), EntityType.DOB, m.group()))

        return found

    def _ent(
        self, start: int, end: int, etype: EntityType, value: str, conf: float = 1.0
    ) -> Entity:
        return Entity(start=start, end=end, type=etype, value=value,
                      confidence=conf, source=self.name)


# Статический типчек: RegexDetector соответствует протоколу Detector.
_: Detector = RegexDetector()
