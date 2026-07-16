"""Контрольные суммы для структурированных идентификаторов РФ.

Валидация резко снижает ложные срабатывания: случайная последовательность из
10 цифр почти никогда не пройдёт проверку ИНН, а из 13 — ОГРН. Это разница
между «качественным обезличиванием» и regex-поделкой, которая маскирует любое
число.
"""

from __future__ import annotations


def _digits(value: str) -> list[int]:
    return [int(c) for c in value if c.isdigit()]


def valid_inn(value: str) -> bool:
    """ИНН: 10 цифр (юрлицо) или 12 (физлицо/ИП), проверка контрольных разрядов."""
    d = _digits(value)
    if len(d) == 10:
        coef = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        check = sum(c * x for c, x in zip(coef, d)) % 11 % 10
        return check == d[9]
    if len(d) == 12:
        coef1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coef2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n11 = sum(c * x for c, x in zip(coef1, d)) % 11 % 10
        n12 = sum(c * x for c, x in zip(coef2, d)) % 11 % 10
        return n11 == d[10] and n12 == d[11]
    return False


def valid_snils(value: str) -> bool:
    """СНИЛС: 11 цифр, последние 2 — контрольное число."""
    d = _digits(value)
    if len(d) != 11:
        return False
    body = d[:9]
    control = d[9] * 10 + d[10]
    s = sum(num * (9 - i) for i, num in enumerate(body))
    if s < 100:
        expected = s
    elif s in (100, 101):
        expected = 0
    else:
        expected = s % 101
        if expected in (100, 101):
            expected = 0
    return expected == control


def valid_ogrn(value: str) -> bool:
    """ОГРН (13 цифр) или ОГРНИП (15 цифр) с контрольным разрядом."""
    d = _digits(value)
    if len(d) == 13:
        num = int("".join(map(str, d[:12])))
        return num % 11 % 10 == d[12]
    if len(d) == 15:
        num = int("".join(map(str, d[:14])))
        return num % 13 % 10 == d[14]
    return False


def valid_luhn(value: str) -> bool:
    """Алгоритм Луна для номеров банковских карт."""
    d = _digits(value)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    parity = len(d) % 2
    for i, num in enumerate(d):
        if i % 2 == parity:
            num *= 2
            if num > 9:
                num -= 9
        total += num
    return total % 10 == 0
