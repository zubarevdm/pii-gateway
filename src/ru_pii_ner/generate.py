"""Template-based synthetic dataset for training.

Шаблоны нужны только для обучающей выборки: тест из них собирать нельзя,
иначе метрика покажет, как модель выучила шаблоны. Тестовый набор размечается
вручную на текстах другого происхождения (см. docs/annotation.md).

    python -m ru_pii_ner.generate -n 5000 -o data/synthetic/train.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

from . import fakes
from .schema import Example, Span

_SLOT = re.compile(r"\{([A-Z0-9_]+)\}")
_MASKABLE = {"INN", "SNILS", "PASSPORT", "PHONE", "CARD", "ACCOUNT"}
MASK_RATE = 0.1

TEMPLATES = [
    # тикеты поддержки
    "Здравствуйте! Меня зовут {PER}, не могу войти в личный кабинет. Номер для связи {PHONE}.",
    "Добрый день. Заказ {ORDER} так и не пришёл. Адрес доставки: {ADDRESS}. Получатель {PER}, тел. {PHONE}",
    "Прошу вернуть деньги за заказ {ORDER} на карту {CARD}. {PER}",
    "Оператор {FIRST} обещал(а) перезвонить ещё вчера, но звонка не было. Мой номер {PHONE}",
    "Не приходит код подтверждения на {EMAIL}. Логин {NICK}. Помогите, пожалуйста",
    "Списали деньги дважды с карты {CARD}, хотя я оплачивал один раз. Прошу разобраться. {PER}, {EMAIL}",
    # чаты и мессенджеры
    "{FIRST}, добрый день! Скиньте, пожалуйста, ваш СНИЛС, нужен для договора",
    "ок, мой снилс {SNILS}, паспорт {PASSPORT}",
    "Передай {LAST_DAT}, что встреча переносится на {DATE}",
    "Пишите в телегу {NICK} или на почту {EMAIL}",
    "{FIRST}, привет) это {PER2}, пишу по объявлению про квартиру на {ADDRESS}",
    "Позвоните {LAST_DAT} по номеру {PHONE}, он в курсе",
    # анкеты и формы
    "ФИО: {PER}\nДата рождения: {BIRTHDATE}\nПаспорт: {PASSPORT}\nАдрес регистрации: {ADDRESS}\nТелефон: {PHONE}",
    "Анкета соискателя. {PER}, {BIRTHDATE} г.р. Контакты: {PHONE}, {EMAIL}. ИНН {INN}, СНИЛС {SNILS}",
    "Заявитель: {PER}\nПаспорт серия {PASSPORT_SER}, номер {PASSPORT_NUM}\nПроживает: {ADDRESS}",
    "Реквизиты для перечисления зарплаты: получатель {PER}, счёт {ACCOUNT}, БИК {BIK}",
    # деловые письма и заявления
    "Генеральному директору {ORG}\nот {PER}, проживающего по адресу: {ADDRESS}\n\nЗАЯВЛЕНИЕ\nПрошу предоставить отпуск с {DATE}.",
    "Уважаемый {PER_FM}! Направляем договор с {ORG} (ИНН {ORG_INN}). Вопросы можно задать по телефону {HOTLINE}.",
    "Договор заключён между {ORG} и ИП {PER} (ИНН {INN}). Оплата по счёту {ACCOUNT}.",
    "Прошу учесть мой ИНН {INN10P} при оформлении вычета. {PER}",
    "Направляю копию паспорта {PASSPORT} и СНИЛС {SNILS}. С уважением, {PER}, {PHONE}",
    # банк, медицина, доставка
    "Клиент {PER}, {BIRTHDATE}, обратился с жалобой на блокировку карты {CARD}. Связаться: {PHONE}",
    "Запись к терапевту: пациент {PER}, дата рождения {BIRTHDATE}, полис оформлен {DATE}",
    "Курьер не смог дозвониться до получателя. Адрес {ADDRESS}, получатель {FIRST}, телефон {PHONE}",
    # без персональных данных: учат модель не срабатывать на похожее
    "Горячая линия {ORG} работает круглосуточно: {HOTLINE}, почта {ORG_EMAIL}",
    "Реквизиты поставщика: {ORG}, ИНН {ORG_INN}, БИК {BIK}. Заказ {ORDER} от {DATE}.",
    "Филиал в городе {CITY} откроется {DATE}. Есть надежда, что сроки не сдвинутся.",
    "Номер заказа {ORDER}, сумма к оплате {SUM} руб. Вопросы: {ORG_EMAIL}",
    "Менеджер свяжется с вами в течение дня. Наш адрес для корреспонденции: {ORG_EMAIL}",
]

_ORGS = ["ООО «Ромашка»", "АО «СеверСталь-Логистик»", "ООО «ТехноСервис»",
         "ПАО «Восток Банк»", "ООО «Дом Мебели»", "АО «Городские сети»"]
_CITIES = ["Казани", "Самаре", "Екатеринбурге", "Твери", "Новосибирске", "Перми"]


class _Context:
    """Значения одного примера: один и тот же человек упоминается согласованно."""

    def __init__(self, rng: random.Random, template: str):
        self.rng = rng
        self.p = fakes.Person.random(rng)
        if "{LAST_DAT}" in template:
            while self.p.last_dative() is None:
                self.p = fakes.Person.random(rng)
        self.p2 = fakes.Person.random(rng)
        self.passport = fakes.passport_parts(rng)

    def fill(self, slot: str) -> tuple[str, str | None]:
        r, p = self.rng, self.p
        table = {
            "PER": lambda: (p.full(r), "PER"),
            "PER2": lambda: (self.p2.full(r), "PER"),
            "PER_FM": lambda: (p.first_middle(), "PER"),
            "FIRST": lambda: (p.first, "PER"),
            "LAST_DAT": lambda: (p.last_dative(), "PER"),
            "NICK": lambda: (p.nick(r), "NICK"),
            "EMAIL": lambda: (p.email(r), "EMAIL"),
            "PHONE": lambda: (fakes.phone(r), "PHONE"),
            "INN": lambda: (fakes.inn12(r), "INN"),
            "INN10P": lambda: (fakes.inn10(r), "INN"),
            "SNILS": lambda: (fakes.snils(r), "SNILS"),
            "PASSPORT": lambda: (fakes.passport(r), "PASSPORT"),
            "PASSPORT_SER": lambda: (self.passport[0], "PASSPORT"),
            "PASSPORT_NUM": lambda: (self.passport[1], "PASSPORT"),
            "ADDRESS": lambda: (fakes.address(r), "ADDRESS"),
            "BIRTHDATE": lambda: (fakes.birthdate(r), "BIRTHDATE"),
            "CARD": lambda: (fakes.card(r), "CARD"),
            "ACCOUNT": lambda: (fakes.account(r), "ACCOUNT"),
            # не персональные данные
            "ORG": lambda: (r.choice(_ORGS), None),
            "ORG_INN": lambda: (fakes.inn10(r), None),
            "ORG_EMAIL": lambda: (r.choice(["info", "support", "hr", "office"]) + "@"
                                  + r.choice(["romashka.ru", "techservice.ru", "vostok-bank.ru"]), None),
            "HOTLINE": lambda: (f"8-800-{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(10, 99)}", None),
            "BIK": lambda: (f"04{r.randint(4000000, 5999999)}", None),
            "ORDER": lambda: (f"№{r.randint(10**6, 10**8)}", None),
            "DATE": lambda: (f"{r.randint(1, 28):02d}.{r.randint(1, 12):02d}.2026", None),
            "CITY": lambda: (r.choice(_CITIES), None),
            "SUM": lambda: (f"{r.randint(300, 90000)}", None),
        }
        value, label = table[slot]()
        if label in _MASKABLE and slot not in ("PASSPORT_SER", "PASSPORT_NUM") and r.random() < MASK_RATE:
            value = fakes.mask(value, r)
        return value, label


def render(template: str, rng: random.Random, ex_id: str) -> Example:
    ctx = _Context(rng, template)
    text, spans, pos = [], [], 0
    for m in _SLOT.finditer(template):
        text.append(template[pos:m.start()])
        value, label = ctx.fill(m.group(1))
        start = sum(map(len, text))
        text.append(value)
        if label:
            spans.append(Span(start=start, end=start + len(value), label=label))
        pos = m.end()
    text.append(template[pos:])
    return Example(id=ex_id, text="".join(text), spans=spans, source="synthetic")


def generate(n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    fakes.seed(seed)
    return [render(rng.choice(TEMPLATES), rng, f"syn-{i:05d}") for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5000)
    ap.add_argument("-o", type=Path, default=Path("data/synthetic/train.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.o.parent.mkdir(parents=True, exist_ok=True)
    with args.o.open("w", encoding="utf-8") as f:
        for ex in generate(args.n, args.seed):
            f.write(json.dumps(ex.model_dump(mode="json"), ensure_ascii=False) + "\n")
    print(f"{args.n} examples -> {args.o}")


if __name__ == "__main__":
    main()
