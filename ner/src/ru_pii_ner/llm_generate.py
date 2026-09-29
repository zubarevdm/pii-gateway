"""Diverse training texts written and annotated by GigaChat.

Шаблоны дают чистую, но однообразную разметку. Здесь LLM пишет тексты разных
жанров сразу с инлайн-разметкой `[значение|МЕТКА]`; границы потом приводятся
к гайду тем же кодом, что и тест. Разметка шумнее ручной — это обучающие данные,
не эталон. Тексты, похожие на тестовые, отбрасываются.

    python -m ru_pii_ner.llm_generate -n 1500   # -> data/llm/train.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

from .adjudicate import normalize
from .llm_annotate import GigaChat, _load_key
from .schema import Example
from .testset import parse

OUT = Path("data/llm/train.jsonl")

GENRES = [
    "обращение в поддержку интернет-магазина", "сообщение в домовом чате", "переписка в мессенджере с другом",
    "заявление в отдел кадров", "резюме соискателя", "жалоба в банк", "запись к врачу", "выписка из медкарты",
    "объявление на Авито", "письмо от госуслуг", "договор аренды", "сообщение курьеру", "отзыв о сервисе",
    "пост в соцсети", "заявка на кредит", "служебная записка", "уведомление о штрафе", "анкета для визы",
    "сообщение в школьном чате родителей", "переписка с арендодателем", "обращение в управляющую компанию",
    "страховой случай ОСАГО", "письмо поставщику с реквизитами", "чат поддержки мобильного оператора",
]
STYLES = ["официально", "разговорно, с сокращениями", "с опечатками и без заглавных букв", "коротко, одной-двумя фразами",
          "развёрнуто, 4–6 предложений", "сухо, как форма или шаблон документа"]

PROMPT = """Напиши 5 разных реалистичных текстов. Жанр: {genre}. Стиль: {style}.
Персональные данные выдумай. Каждую сущность размечай прямо в тексте как [значение|МЕТКА].
Метки: PER (ФИО, имя или фамилия человека в любом падеже), NICK (ник, логин), INN (ИНН физлица), SNILS,
PASSPORT (серия и/или номер), PHONE, EMAIL, ADDRESS (адрес с улицей и домом), BIRTHDATE (дата рождения или возраст:
«32 года»), CARD (номер карты, в т.ч. с маской ****), ACCOUNT (счёт физлица), DOC_ID (полис ОМС/ОСАГО, госномер,
права, больничный), ORG_ID (ИНН/ОГРН/КПП/счёт/email/телефон организации).
Не размечай: названия организаций, города без улицы, должности, обычные даты, номера заказов.
Слова-маркеры («тел.», «ИНН», «паспорт») в скобки не включай.
Примерно в одном тексте из пяти персональных данных быть не должно.
Разделяй тексты строкой ---. Больше ничего не пиши."""


def _too_similar(text: str, test_texts: list[str]) -> bool:
    words = set(re.findall(r"\w{4,}", text.lower()))
    for t in test_texts:
        tw = set(re.findall(r"\w{4,}", t.lower()))
        if tw and len(words & tw) / len(tw | words) > 0.5:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1500, help="сколько текстов собрать")
    ap.add_argument("--env", type=Path, default=Path("../gigachat.env"))
    ap.add_argument("--model", default="GigaChat-2-Max")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    client = GigaChat(_load_key(args.env), args.model)
    test_texts = [json.loads(l)["text"] for l in Path("data/test/test.jsonl").open(encoding="utf-8")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in OUT.open(encoding="utf-8")) if OUT.exists() else 0
    stats = {"ok": have, "bad_markup": 0, "similar_to_test": 0}
    with OUT.open("a", encoding="utf-8") as f:
        call = 0
        while stats["ok"] < args.n:
            genre, style = rng.choice(GENRES), rng.choice(STYLES)
            raw = client.complete(PROMPT.format(genre=genre, style=style), system=None, max_tokens=2500, temperature=0.9)
            call += 1
            for chunk in (c.strip() for c in raw.replace("\r\n", "\n").split("\n---")):
                chunk = re.sub(r"^\s*(\d+[.)]|Текст \d+:?)\s*", "", chunk.strip("- \n"))
                if len(chunk) < 20:
                    continue
                try:
                    ex = parse(chunk, f"llm-{stats['ok']:05d}", rng)
                except Exception:
                    stats["bad_markup"] += 1
                    continue
                if _too_similar(ex.text, test_texts):
                    stats["similar_to_test"] += 1
                    continue
                spans = [n for s in ex.spans if (n := normalize(ex.text, s))]
                out = Example(id=ex.id, text=ex.text, spans=spans, source=f"gigachat:{genre}")
                f.write(json.dumps(out.model_dump(mode="json"), ensure_ascii=False) + "\n")
                f.flush()
                stats["ok"] += 1
            if call % 20 == 0:
                print(stats, flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
