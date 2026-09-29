"""Dataset format and entity labels. See docs/annotation.md for span rules."""
from enum import StrEnum

from pydantic import BaseModel, model_validator


class Label(StrEnum):
    PER = "PER"              # ФИО физлица
    NICK = "NICK"            # никнейм/логин; в сводной метрике объединяется с PER
    INN = "INN"              # ИНН (10 или 12 цифр)
    SNILS = "SNILS"
    PASSPORT = "PASSPORT"    # серия и номер паспорта РФ
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"      # адрес физлица
    BIRTHDATE = "BIRTHDATE"
    CARD = "CARD"            # номер банковской карты
    ACCOUNT = "ACCOUNT"      # расчётный/лицевой счёт
    DOC_ID = "DOC_ID"        # полис ОМС/ОСАГО, госномер, ВУ, больничный
    ORG_ID = "ORG_ID"        # реквизиты и контакты организаций; не ПД, отдельная политика в шлюзе


class Span(BaseModel):
    start: int
    end: int
    label: Label


class Example(BaseModel):
    id: str
    text: str
    spans: list[Span]
    source: str  # synthetic | llm | real, чтобы считать метрики по срезам

    @model_validator(mode="after")
    def _check_spans(self) -> "Example":
        prev_end = 0
        for s in sorted(self.spans, key=lambda s: s.start):
            if not (0 <= s.start < s.end <= len(self.text)):
                raise ValueError(f"{self.id}: span {s} out of text bounds")
            if s.start < prev_end:
                raise ValueError(f"{self.id}: overlapping spans at {s.start}")
            if self.text[s.start].isspace() or self.text[s.end - 1].isspace():
                raise ValueError(f"{self.id}: span {s} has surrounding whitespace")
            prev_end = s.end
        return self
