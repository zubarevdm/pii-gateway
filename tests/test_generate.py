import random
import re

import pytest

from ru_pii_ner import fakes
from ru_pii_ner.generate import TEMPLATES, generate, render
from ru_pii_ner.schema import Example


def _inn_ok(inn: str) -> bool:
    d = list(map(int, inn))
    if len(d) == 10:
        return fakes._checksum(d[:9], [2, 4, 10, 3, 5, 9, 4, 6, 8]) == d[9]
    return (fakes._checksum(d[:10], [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == d[10]
            and fakes._checksum(d[:11], [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == d[11])


def test_known_valid_values():
    assert _inn_ok("7707083893")      # ИНН Сбербанка
    assert _inn_ok("500100732259")
    assert fakes.snils_checksum("112233445") == 95
    assert fakes.luhn_digit("7992739871") == 3


@pytest.mark.parametrize("seed", range(50))
def test_generated_checksums(seed):
    rng = random.Random(seed)
    assert _inn_ok(fakes.inn12(rng))
    assert _inn_ok(fakes.inn10(rng))
    s = re.sub(r"\D", "", fakes.snils(rng))
    assert fakes.snils_checksum(s[:9]) == int(s[9:])
    c = re.sub(r"\D", "", fakes.card(rng))
    assert fakes.luhn_digit(c[:-1]) == int(c[-1])


def test_mask_keeps_format_and_tail():
    rng = random.Random(1)
    for _ in range(100):
        v = fakes.phone(rng)
        m = fakes.mask(v, rng)
        assert len(m) == len(v) and "*" in m
        assert re.sub(r"[\d*]", "", m) == re.sub(r"\d", "", v)
        assert m[-1].isdigit()


def test_every_template_renders():
    rng = random.Random(0)
    for i, t in enumerate(TEMPLATES):
        ex = render(t, rng, f"t{i}")
        assert "{" not in ex.text


def test_spans_match_labels():
    for ex in generate(500, seed=7):
        Example.model_validate(ex.model_dump())
        for s in ex.spans:
            value = ex.text[s.start:s.end]
            if s.label in ("PHONE", "CARD", "SNILS", "INN", "ACCOUNT"):
                assert re.fullmatch(r"[\d*+() \-№]+", value), (s.label, value)
            if s.label == "EMAIL":
                assert "@" in value


def test_generation_is_deterministic():
    assert generate(20, seed=3) == generate(20, seed=3)
