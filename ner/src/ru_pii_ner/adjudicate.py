"""Compare two independent annotations and resolve disagreements by hand.

    python -m ru_pii_ner.adjudicate agreement   # согласие разметчиков по меткам
    python -m ru_pii_ner.adjudicate questions   # -> data/test/adjudication.md (вопросы A/B)
    python -m ru_pii_ner.adjudicate resolve "1A 2B 3N ..."   # -> data/test/test.jsonl

Совпавшие спаны принимаются автоматически. Каждое расхождение превращается в
вопрос с вариантами A и B; кто из разметчиков стоит под буквой, выбирается
случайно, чтобы решение не зависело от авторства. N — оба варианта неверны,
пример уходит в журнал на ручную правку.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from .schema import Example, Span

DIR = Path("data/test")
FIRST, SECOND = DIR / "preannotated.jsonl", DIR / "gigachat.jsonl"
QUESTIONS = DIR / "adjudication.json"


_LEAD = re.compile(r"^(?:ИНН|ОГРН|КПП|ОКПО|СНИЛС|ИП|паспорт(?:а)?|серия|номер|№|тел\.?|телефон|e-?mail)[\s:]*", re.I)
_TAIL = re.compile(r"[\s,;:]*(?:г\.\s?р\.?)$")


def normalize(text: str, span: Span) -> Span | None:
    """Привести границы к гайду: без слов-маркеров и краевой пунктуации, с точкой инициала и «г.»."""
    start, end = span.start, span.end
    for _ in range(3):  # «паспорт серия 4012» снимается в несколько проходов
        m = _LEAD.match(text[start:end])
        if m and m.end() < end - start:
            start += m.end()
    m = _TAIL.search(text[start:end])
    if m and m.start() > 0:
        end = start + m.start()
    while end > start and text[end - 1] in " .,;:«»\"'()":
        end -= 1
    while start < end and text[start] in " .,;:«»\"'()":
        start += 1
    last = re.findall(r"\w+", text[start:end])
    if last and text[end:end + 1] == "." and ((len(last[-1]) == 1 and last[-1].isalpha()) or last[-1] == "г"):
        end += 1  # инициал «Т.» или год «1998 г.»
    return Span(start=start, end=end, label=span.label) if end > start else None


def _load(path: Path) -> dict[str, Example]:
    out = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        spans = [n for s in row["spans"] if (n := normalize(row["text"], Span(**s)))]
        out[row["id"]] = Example(id=row["id"], text=row["text"], spans=spans, source=row["source"])
    return out


def _key(s: Span) -> tuple:
    return s.start, s.end, s.label.value


def agreement() -> None:
    a, b = _load(FIRST), _load(SECOND)
    tp, fa, fb = Counter(), Counter(), Counter()
    for i in a.keys() & b.keys():
        ka, kb = {_key(s) for s in a[i].spans}, {_key(s) for s in b[i].spans}
        for k in ka & kb: tp[k[2]] += 1
        for k in ka - kb: fa[k[2]] += 1
        for k in kb - ka: fb[k[2]] += 1
    print(f"{'label':10} {'agree':>5} {'only1':>5} {'only2':>5} {'F1':>6}")
    for label in sorted(set(tp) | set(fa) | set(fb)):
        f1 = 2 * tp[label] / (2 * tp[label] + fa[label] + fb[label])
        print(f"{label:10} {tp[label]:5} {fa[label]:5} {fb[label]:5} {f1:6.2f}")
    t, x, y = sum(tp.values()), sum(fa.values()), sum(fb.values())
    print(f"{'ALL':10} {t:5} {x:5} {y:5} {2 * t / (2 * t + x + y):6.2f}")
    same = sum({_key(s) for s in a[i].spans} == {_key(s) for s in b[i].spans} for i in a.keys() & b.keys())
    print(f"examples fully agreed: {same}/{len(a.keys() & b.keys())}")


def _groups(a: list[Span], b: list[Span]) -> list[tuple[list[Span], list[Span]]]:
    """Сгруппировать несовпавшие спаны в пересекающиеся кластеры: одно расхождение — один вопрос."""
    items = [(s, 0) for s in a] + [(s, 1) for s in b]
    items.sort(key=lambda x: x[0].start)
    groups, cur, cur_end = [], [], -1
    for s, who in items:
        if cur and s.start >= cur_end:
            groups.append(cur); cur = []
        cur.append((s, who)); cur_end = max(cur_end, s.end) if cur[:-1] else s.end
    if cur:
        groups.append(cur)
    return [([s for s, w in g if w == 0], [s for s, w in g if w == 1]) for g in groups]


def _render(text: str, spans: list[Span], lo: int, hi: int) -> str:
    if not spans:
        return "ничего не размечать"
    return "; ".join(f"{s.label.value}: «{text[s.start:s.end]}»" for s in spans)


def questions(seed: int = 7) -> None:
    a, b = _load(FIRST), _load(SECOND)
    rng = random.Random(seed)
    qs, lines = [], []
    for i in sorted(a.keys() & b.keys()):
        ka, kb = {_key(s) for s in a[i].spans}, {_key(s) for s in b[i].spans}
        only_a = [s for s in a[i].spans if _key(s) not in kb]
        only_b = [s for s in b[i].spans if _key(s) not in ka]
        text = a[i].text
        for ga, gb in _groups(only_a, only_b):
            lo = min(s.start for s in ga + gb); hi = max(s.end for s in ga + gb)
            swap = rng.random() < 0.5
            opt = {"A": gb if swap else ga, "B": ga if swap else gb}
            n = len(qs) + 1
            qs.append({"n": n, "id": i, "A": [s.model_dump(mode="json") for s in opt["A"]],
                       "B": [s.model_dump(mode="json") for s in opt["B"]], "A_is": "second" if swap else "first"})
            ctx = text[max(0, lo - 60):hi + 60].replace("\n", " ")
            ctx = ("…" if lo > 60 else "") + ctx + ("…" if hi + 60 < len(text) else "")
            lines.append(f"**{n}.** {ctx}\n   A — {_render(text, opt['A'], lo, hi)}\n   B — {_render(text, opt['B'], lo, hi)}\n")
    QUESTIONS.write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    (DIR / "adjudication.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(qs)} questions -> {DIR / 'adjudication.md'}")


def _apply_corrections(examples: dict[str, Example], final: dict[str, list[Span]]) -> None:
    """Ручные правки поверх сверки: решения, принятые после неё (data/test/corrections.jsonl)."""
    path = DIR / "corrections.jsonl"
    if not path.exists():
        return
    for line in path.open(encoding="utf-8"):
        fix = json.loads(line)
        text = examples[fix["id"]].text
        for item in fix.get("add", []):
            anchor = item.get("after", "")
            start = text.find(anchor + item["text"])
            if start < 0:
                raise SystemExit(f"{fix['id']}: не найдено «{anchor}{item['text']}»")
            start += len(anchor)
            span = Span(start=start, end=start + len(item["text"]), label=item["label"])
            final[fix["id"]] = [s for s in final[fix["id"]] if s.end <= span.start or s.start >= span.end] + [span]


def resolve(answers: str) -> None:
    a = _load(FIRST)
    qs = {q["n"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
    picks = {int(n): c.upper() for n, c in re.findall(r"(\d+)\s*([ABNabnАВав])", answers)}
    picks = {n: {"А": "A", "В": "B"}.get(c, c) for n, c in picks.items()}  # кириллица с клавиатуры
    missing = sorted(set(qs) - set(picks))
    if missing:
        raise SystemExit(f"нет ответа на вопросы: {missing}")
    b = _load(SECOND)
    final, flagged = {}, []
    for i, ex in a.items():
        agreed = [s for s in ex.spans if _key(s) in {_key(x) for x in b[i].spans}]
        final[i] = agreed
    for n, q in qs.items():
        if picks[n] == "N" and (not q["A"] or not q["B"]):
            picks[n] = "A" if not q["A"] else "B"  # «оба неверны» при пустом варианте = «ничего не размечать»
        if picks[n] == "N":
            flagged.append(q["id"]); continue
        final[q["id"]] += [Span(**s) for s in q[picks[n]]]
    _apply_corrections(a, final)
    with (DIR / "test.jsonl").open("w", encoding="utf-8") as f:
        for i, ex in a.items():
            out = Example(id=i, text=ex.text, spans=sorted(final[i], key=lambda s: s.start), source="adjudicated")
            f.write(json.dumps(out.model_dump(mode="json"), ensure_ascii=False) + "\n")
    stats = Counter(("first" if (picks[n] == "A") == (q["A_is"] == "first") else "second") for n, q in qs.items() if picks[n] != "N")
    print(f"test.jsonl written; wins: {dict(stats)}; both wrong: {len(flagged)} -> {sorted(set(flagged))}")
    (DIR / "adjudication_log.json").write_text(json.dumps(
        {"answers": picks, "wins": stats, "both_wrong": flagged}, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("agreement"); sub.add_parser("questions")
    r = sub.add_parser("resolve"); r.add_argument("answers")
    args = ap.parse_args()
    {"agreement": agreement, "questions": questions}.get(args.cmd, lambda: resolve(args.answers))()


if __name__ == "__main__":
    main()
