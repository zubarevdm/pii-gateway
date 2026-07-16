"""Базовые типы детекции ПД.

Detector — это единый интерфейс для любого слоя детекции (regex, NER, будущая
LLM-модель). Каждый детектор получает текст и возвращает список найденных
сущностей `Entity` с позициями. Pipeline объединяет результаты и разрешает
пересечения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class EntityType(str, Enum):
    """Категории персональных данных.

    Порядок важен только для человекочитаемости; приоритет при разрешении
    пересечений задаётся отдельно в pipeline (структурированное > NER).
    """

    # --- структурированные (regex/словари, ~100% точность) ---
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    INN = "INN"            # ИНН (10 или 12 цифр)
    SNILS = "SNILS"        # СНИЛС
    PASSPORT = "PASSPORT"  # серия+номер паспорта РФ
    CARD = "CARD"          # номер банковской карты
    ACCOUNT = "ACCOUNT"    # расчётный/лицевой счёт (20 цифр)
    OGRN = "OGRN"          # ОГРН / ОГРНИП
    CAR_PLATE = "CAR_PLATE"  # госномер авто
    IP = "IP"              # IP-адрес
    DOB = "DOB"            # дата рождения

    # --- свободный текст (NER) ---
    PERSON = "PERSON"      # ФИО
    ORG = "ORG"            # организация
    LOCATION = "LOCATION"  # адрес / населённый пункт

    def placeholder_base(self) -> str:
        """Базовое имя плейсхолдера, например PERSON -> [PERSON_1]."""
        return self.value


@dataclass(frozen=True, slots=True)
class Entity:
    """Найденная сущность ПД в исходном тексте.

    start/end — индексы среза исходной строки (text[start:end] == value).
    confidence — уверенность детектора (1.0 для regex, <1 для NER).
    source — какой детектор нашёл (для аудита и отладки).
    """

    start: int
    end: int
    type: EntityType
    value: str
    confidence: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Некорректный диапазон сущности: {self.start}:{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end


class Detector(Protocol):
    """Контракт слоя детекции. Любая реализация (regex, Natasha, LLM) подходит."""

    name: str

    def detect(self, text: str) -> list[Entity]:
        ...


@dataclass
class DetectionResult:
    """Результат прохода pipeline по одному тексту."""

    text: str
    entities: list[Entity] = field(default_factory=list)
