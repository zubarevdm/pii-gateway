"""Тесты слоя 1: структурированные ПД и контрольные суммы."""

from app.detectors.base import EntityType
from app.detectors.regex_detector import RegexDetector
from app.detectors.validators import valid_inn, valid_ogrn, valid_snils, valid_luhn

det = RegexDetector()


def _types(text: str) -> set[EntityType]:
    return {e.type for e in det.detect(text)}


def test_email_and_phone():
    t = "Пишите на ivan.petrov@example.com или звоните +7 (900) 123-45-67."
    types = _types(t)
    assert EntityType.EMAIL in types
    assert EntityType.PHONE in types


def test_phone_variants():
    assert _types("8 900 123 45 67") == {EntityType.PHONE}
    assert _types("+79001234567") == {EntityType.PHONE}


def test_valid_inn_checksum():
    # Валидные ИНН (контрольные суммы сходятся).
    assert valid_inn("7707083893")    # 10-значный (Сбербанк, публично известный)
    assert valid_inn("500100732259")  # 12-значный пример
    assert not valid_inn("1234567890")


def test_snils_checksum():
    assert valid_snils("112-233-445 95")
    assert not valid_snils("112-233-445 96")


def test_ogrn_checksum():
    assert valid_ogrn("1027700132195")  # 13-значный пример
    assert not valid_ogrn("1027700132190")


def test_luhn_card():
    assert valid_luhn("4276 3800 1234 5675") or valid_luhn("4111111111111111")
    assert not valid_luhn("1234567812345678")


def test_inn_detected_only_if_valid():
    # Случайные 10 цифр не должны маскироваться как ИНН.
    assert EntityType.INN not in _types("номер заказа 1234567890")
    assert EntityType.INN in _types("ИНН 7707083893")


def test_no_false_positive_on_plain_text():
    assert det.detect("Сегодня хорошая погода, встретимся в офисе.") == []
