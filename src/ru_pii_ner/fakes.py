"""Generators of realistic Russian PII values with valid checksums."""
import random
from dataclasses import dataclass

from faker import Faker

_fake = Faker("ru_RU")

_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
})

_MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
               "августа", "сентября", "октября", "ноября", "декабря"]


def seed(n: int) -> None:
    Faker.seed(n)


def translit(s: str) -> str:
    return s.lower().translate(_TRANSLIT)


def _digits(rng: random.Random, n: int) -> list[int]:
    return [rng.randint(0, 9) for _ in range(n)]


def _checksum(digits: list[int], weights: list[int]) -> int:
    return sum(d * w for d, w in zip(digits, weights)) % 11 % 10


def inn12(rng: random.Random) -> str:
    d = [rng.randint(0, 9), rng.randint(1, 9)] + _digits(rng, 8)
    d.append(_checksum(d, [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]))
    d.append(_checksum(d, [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]))
    return "".join(map(str, d))


def inn10(rng: random.Random) -> str:
    d = [rng.randint(0, 9), rng.randint(1, 9)] + _digits(rng, 7)
    d.append(_checksum(d, [2, 4, 10, 3, 5, 9, 4, 6, 8]))
    return "".join(map(str, d))


def snils_checksum(nine: str) -> int:
    s = sum(int(c) * (9 - i) for i, c in enumerate(nine))
    if s < 100:
        return s
    if s in (100, 101):
        return 0
    c = s % 101
    return 0 if c == 100 else c


def snils(rng: random.Random) -> str:
    nine = str(rng.randint(1_001_999, 999_999_999)).zfill(9)
    c = f"{snils_checksum(nine):02d}"
    if rng.random() < 0.8:
        return f"{nine[:3]}-{nine[3:6]}-{nine[6:]} {c}"
    return nine + c


def passport_parts(rng: random.Random) -> tuple[str, str]:
    series = f"{rng.randint(1, 99):02d}{rng.randint(0, 25):02d}"
    return series, f"{rng.randint(100000, 999999)}"


def passport(rng: random.Random) -> str:
    s, n = passport_parts(rng)
    return rng.choice([f"{s[:2]} {s[2:]} {n}", f"{s} {n}", f"{s}{n}", f"{s} № {n}"])


def phone(rng: random.Random) -> str:
    code = f"9{rng.randint(0, 99):02d}"
    a, b, c = rng.randint(100, 999), rng.randint(0, 99), rng.randint(0, 99)
    return rng.choice([
        f"+7 ({code}) {a}-{b:02d}-{c:02d}",
        f"+7 {code} {a}-{b:02d}-{c:02d}",
        f"8 {code} {a} {b:02d} {c:02d}",
        f"8-{code}-{a}-{b:02d}-{c:02d}",
        f"+7{code}{a}{b:02d}{c:02d}",
        f"8{code}{a}{b:02d}{c:02d}",
    ])


def luhn_digit(body: str) -> int:
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def card(rng: random.Random) -> str:
    prefix = rng.choice(["2200", "2201", "2202", "2204", "4276", "5536"])
    body = prefix + "".join(map(str, _digits(rng, 11)))
    num = body + str(luhn_digit(body))
    if rng.random() < 0.7:
        return " ".join(num[i:i + 4] for i in range(0, 16, 4))
    return num


def account(rng: random.Random) -> str:
    # 408 17 810: счёт физлица-резидента в рублях. Контрольный разряд зависит
    # от БИК, поэтому здесь он случайный.
    return "40817810" + "".join(map(str, _digits(rng, 12)))


def birthdate(rng: random.Random) -> str:
    y, m, d = rng.randint(1950, 2006), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([
        f"{d:02d}.{m:02d}.{y}",
        f"{d} {_MONTHS_GEN[m - 1]} {y} г.",
        f"{d} {_MONTHS_GEN[m - 1]} {y} года",
        f"{d:02d}.{m:02d}.{y % 100:02d}",
    ])


def address(rng: random.Random) -> str:
    street = rng.choice(["ул.", "улица", "пр-т", "проспект", "пер.", "бульвар"])
    parts = [f"{street} {_fake.street_title()}", f"д. {rng.randint(1, 150)}"]
    if rng.random() < 0.3:
        parts[-1] = f"{rng.randint(1, 150)}"
    if rng.random() < 0.2:
        parts.append(f"корп. {rng.randint(1, 5)}")
    if rng.random() < 0.7:
        parts.append(f"кв. {rng.randint(1, 400)}")
    if rng.random() < 0.6:
        parts.insert(0, f"г. {_fake.city_name()}")
    if rng.random() < 0.2:
        parts.insert(0, str(rng.randint(101000, 692999)))
    return ", ".join(parts)


def mask(value: str, rng: random.Random) -> str:
    """Hide a run of digits with '*', keeping at least the last 2 visible."""
    idx = [i for i, ch in enumerate(value) if ch.isdigit()]
    keep_tail = rng.choice([2, 4])
    hide = idx[: max(len(idx) - keep_tail, 1)]
    start = rng.randint(0, len(hide) // 2)
    hide = set(hide[start:])
    return "".join("*" if i in hide else ch for i, ch in enumerate(value))


@dataclass
class Person:
    first: str
    middle: str
    last: str
    male: bool

    @classmethod
    def random(cls, rng: random.Random) -> "Person":
        if rng.random() < 0.5:
            return cls(_fake.first_name_male(), _fake.middle_name_male(), _fake.last_name_male(), True)
        return cls(_fake.first_name_female(), _fake.middle_name_female(), _fake.last_name_female(), False)

    def full(self, rng: random.Random) -> str:
        i, o = self.first[0], self.middle[0]
        return rng.choice([
            f"{self.last} {self.first} {self.middle}",
            f"{self.first} {self.middle} {self.last}",
            f"{self.last} {i}. {o}.",
            f"{self.last} {i}.{o}.",
            f"{i}. {o}. {self.last}",
            f"{self.first} {self.last}",
        ])

    def first_middle(self) -> str:
        return f"{self.first} {self.middle}"

    def last_dative(self) -> str | None:
        """Фамилия в дательном падеже для частых окончаний, иначе None."""
        s = self.last
        if self.male and s.endswith(("ов", "ев", "ёв", "ин", "ын")):
            return s + "у"
        if not self.male and s.endswith(("ова", "ева", "ёва", "ина", "ына")):
            return s[:-1] + "ой"
        return None

    def email(self, rng: random.Random) -> str:
        f, l = translit(self.first), translit(self.last)
        local = rng.choice([f"{f}.{l}", f"{f[0]}.{l}", f"{l}{rng.randint(1, 99)}",
                            f"{f}_{l}", f"{l}.{f}{rng.randint(1970, 2005)}"])
        return f"{local}@{rng.choice(['mail.ru', 'yandex.ru', 'gmail.com', 'bk.ru', 'inbox.ru', 'ya.ru'])}"

    def nick(self, rng: random.Random) -> str:
        f, l = translit(self.first), translit(self.last)
        base = rng.choice([f"{f}_{l}", f"{f}{rng.randint(1, 2005)}", f"{l}.{f[0]}", f"{f[0]}{l}"])
        return "@" + base if rng.random() < 0.7 else base
