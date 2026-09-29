"""Second, independent annotation of the test set with GigaChat.

Модель возвращает фрагменты текстом, позиции ищет код: считать символы LLM
не умеют. Ответы кэшируются построчно, поэтому прерванный прогон продолжается
с того же места.

    python -m ru_pii_ner.llm_annotate            # -> data/test/gigachat.jsonl
"""
import argparse
import json
import re
import time
import uuid
import warnings
from pathlib import Path

import httpx

from .schema import Example, Label, Span

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
PRE = Path("data/test/preannotated.jsonl")
OUT = Path("data/test/gigachat.jsonl")

# Цепочка сертификатов НУЦ Минцифры обычно не установлена в системе.
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

SYSTEM = """Ты размечаешь персональные данные в русских текстах для датасета NER.
Верни JSON: {"entities": [{"text": "<точный фрагмент из текста>", "label": "<метка>"}]}
Фрагменты копируй символ в символ, в порядке появления: не меняй падеж, регистр и опечатки. Если размечать нечего, верни {"entities": []}.

Метки:
PER — ФИО, фамилия с инициалами, одиночное имя или фамилия конкретного человека, ФИО внутри «ИП ...».
  Не размечай: исторических и публичных лиц вне адреса («памятник Пушкину»), имена без конкретного
  человека («в классе три Саши»), слова-омонимы не в роли имени («вера в себя»), образцы из инструкций.
NICK — никнейм или логин человека (@anna_petrova, anna1990, t.me/имя).
INN — ИНН физлица (12 цифр; 10 цифр, если из контекста это ИНН человека).
SNILS — СНИЛС. PASSPORT — серия и/или номер паспорта. PHONE — телефон человека, в любой записи, даже словами.
EMAIL — личная почта, в т.ч. именная рабочая и с опечатками. ADDRESS — адрес с улицей и домом.
BIRTHDATE — дата или точный год рождения (не десятилетие, не возраст).
CARD — номер банковской карты, в т.ч. с маской (**** 6789). ACCOUNT — счёт физлица, лицевой счёт.
DOC_ID — полис ОМС/ОСАГО, госномер авто, водительское удостоверение, номер больничного.
ORG_ID — реквизиты и контакты организаций: ИНН/ОГРН/КПП/ОКПО/счёт юрлица, общий email (info@, support@),
  телефон организации (в т.ч. 8-800), аккаунт организации в соцсети.
  Не размечай: БИК, корр. счёт, названия организаций, номера заказов, дел, треков.

Границы: без слов-маркеров («тел.», «ИНН», «паспорт», «МИР»), без кавычек и точки на краях.
Маскированные значения (звёздочки) размечай той же меткой, что и полные.
Если значение разорвано текстом («серия 4510, номер 123456»), это два фрагмента."""


def _load_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and line.split("=", 1)[0].strip().lower() == "authorization key":
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"В {path} нет строки 'Authorization Key = ...'")


class GigaChat:
    def __init__(self, auth_key: str, model: str):
        self.auth_key, self.model = auth_key, model
        self.http = httpx.Client(verify=False, trust_env=False, timeout=90)
        self._token, self._expires = "", 0.0

    def _auth(self) -> str:
        if time.time() < self._expires - 60:
            return self._token
        r = self.http.post(OAUTH_URL, data={"scope": "GIGACHAT_API_PERS"}, headers={
            "Authorization": f"Basic {self.auth_key}", "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
        r.raise_for_status()
        payload = r.json()
        self._token, self._expires = payload["access_token"], payload["expires_at"] / 1000
        return self._token

    def complete(self, text: str, system: str | None = SYSTEM, max_tokens: int = 1500,
                 temperature: float = 0.000001) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
        for attempt in range(4):
            r = self.http.post(API_URL, json=body, headers={"Authorization": f"Bearer {self._auth()}"})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code == 401:
                self._expires = 0
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GigaChat {r.status_code}: {r.text[:200]}")


def parse_entities(raw: str) -> list[dict]:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        ents = json.loads(m.group(0)).get("entities", [])
    except json.JSONDecodeError:
        return []
    return [e for e in ents if isinstance(e, dict) and e.get("text") and e.get("label") in Label.__members__]


_WORD = re.compile(r"\w+", re.U)


def _fuzzy_find(text: str, frag: str) -> tuple[int, int] | None:
    """Найти фрагмент с точностью до окончаний: модель иногда ставит ФИО в именительный падеж."""
    fw = _WORD.findall(frag.lower())
    tw = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]
    if not fw:
        return None

    def close(a: str, b: str) -> bool:
        if len(a) <= 2 or len(b) <= 2:
            return a == b
        k = max(min(len(a), len(b)) - 2, 2)
        return a[:k] == b[:k]

    for i in range(len(tw) - len(fw) + 1):
        if all(close(f, tw[i + j][0]) for j, f in enumerate(fw)):
            end = tw[i + len(fw) - 1][2]
            if len(tw[i + len(fw) - 1][0]) == 1 and text[end:end + 1] == ".":
                end += 1  # точка инициала
            return tw[i][1], end
    return None


def align(text: str, entities: list[dict]) -> tuple[list[Span], list[dict]]:
    """Найти позиции фрагментов по порядку; ненайденные и пересекающиеся вернуть отдельно."""
    spans, missed, cursor = [], [], 0
    for e in entities:
        frag = e["text"].strip(" .,;:«»\"'()")
        start = text.find(frag, cursor)
        if start < 0:
            start = text.find(frag)
        end = start + len(frag)
        if start < 0 and frag:
            found = _fuzzy_find(text, frag)
            start, end = found if found else (-1, -1)
        if start < 0 or not frag:
            missed.append(e)
            continue
        if any(s.start < end and start < s.end for s in spans):
            missed.append(e)
            continue
        spans.append(Span(start=start, end=end, label=Label(e["label"])))
        cursor = end
    return sorted(spans, key=lambda s: s.start), missed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=Path, default=Path("../gigachat.env"))
    ap.add_argument("--model", default="GigaChat-2-Max")
    args = ap.parse_args()
    client = GigaChat(_load_key(args.env), args.model)
    done = {json.loads(l)["id"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    examples = [Example.model_validate_json(l) for l in PRE.open(encoding="utf-8")]
    n_missed = 0
    with OUT.open("a", encoding="utf-8") as f:
        for i, ex in enumerate(e for e in examples if e.id not in done):
            raw = client.complete(ex.text)
            spans, missed = align(ex.text, parse_entities(raw))
            n_missed += len(missed)
            out = Example(id=ex.id, text=ex.text, spans=spans, source="gigachat")
            row = out.model_dump(mode="json") | {"raw": raw, "missed": missed, "model": args.model}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{len(done) + i + 1}/{len(examples)}", flush=True)
    print(f"done; fragments not found in text: {n_missed}")


if __name__ == "__main__":
    main()
