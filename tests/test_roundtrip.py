"""Тесты обратимости: обезличить -> восстановить == исходный смысл.

Проверяем ключевую гарантию продукта:
  - в обезличенном тексте нет исходных ПД;
  - восстановление по Vault возвращает оригинал;
  - одинаковые ПД дают один плейсхолдер (связность для модели).
"""

from app.anonymizer import Anonymizer, Vault
from app.detectors.base import Detector, Entity, EntityType
from app.detectors.pipeline import DetectionPipeline
from app.detectors.regex_detector import RegexDetector


def _anon(detectors: list[Detector]) -> Anonymizer:
    return Anonymizer(DetectionPipeline(detectors))


def test_phone_email_roundtrip():
    anon = _anon([RegexDetector()])
    vault = Vault()
    text = "Клиент ivan@example.com, тел +7 900 123-45-67."
    res = anon.anonymize(text, vault)

    # ПД не должно остаться в обезличенном тексте.
    assert "ivan@example.com" not in res.text
    assert "900" not in res.text
    assert "[EMAIL_1]" in res.text
    assert "[PHONE_1]" in res.text

    # Восстановление возвращает оригинал.
    assert anon.restore(res.text, vault) == text


def test_same_value_same_placeholder():
    anon = _anon([RegexDetector()])
    vault = Vault()
    text = "Звони +79001234567. Повторяю: +7 900 123 45 67 — один и тот же номер."
    res = anon.anonymize(text, vault)
    # Оба упоминания одного номера — один плейсхолдер.
    assert res.text.count("[PHONE_1]") == 2
    assert "[PHONE_2]" not in res.text


class _FakeNER:
    """Игрушечный NER, чтобы тестировать слияние без тяжёлой модели."""

    name = "fake"

    def detect(self, text: str) -> list[Entity]:
        out = []
        i = text.find("Иванов")
        if i != -1:
            out.append(Entity(i, i + 6, EntityType.PERSON, "Иванов",
                              confidence=0.85, source=self.name))
        return out


def test_person_and_structured_together():
    anon = _anon([RegexDetector(), _FakeNER()])
    vault = Vault()
    text = "Иванов, ИНН 7707083893, тел +79001234567."
    res = anon.anonymize(text, vault)
    assert "[PERSON_1]" in res.text
    assert "[INN_1]" in res.text
    assert "[PHONE_1]" in res.text
    assert "Иванов" not in res.text
    assert anon.restore(res.text, vault) == text
