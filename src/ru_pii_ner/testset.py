"""Build the test set from inline-annotated drafts and export it to Label Studio.

Черновики лежат в data/test/drafts/*.txt: примеры разделены строкой `---`,
сущности размечены как `[значение|МЕТКА]`. Вместо значения можно написать
плейсхолдер ($INN12, $INN10, $SNILS, $CARD, $ACCOUNT), тогда подставится
случайное значение с корректной контрольной суммой.

    python -m ru_pii_ner.testset build   # drafts -> data/test/preannotated.jsonl + Label Studio
    python -m ru_pii_ner.testset import data/test/label_studio_export.json  # -> data/test/test.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

from . import fakes
from .schema import Example, Label, Span

DRAFTS = Path("data/test/drafts")
OUT = Path("data/test")
BLIND_SIZE = 50
_MARK = re.compile(r"\[([^\[\]|]+)\|([A-Z]+)\]")
_PLACEHOLDERS = {
    "$INN12": fakes.inn12, "$INN10": fakes.inn10, "$SNILS": fakes.snils,
    "$CARD": fakes.card, "$ACCOUNT": fakes.account,
}


def parse(raw: str, ex_id: str, rng: random.Random) -> Example:
    text, spans, pos = [], [], 0
    for m in _MARK.finditer(raw):
        text.append(raw[pos:m.start()])
        value, label = m.group(1), m.group(2)
        if value in _PLACEHOLDERS:
            value = _PLACEHOLDERS[value](rng)
        start = sum(map(len, text))
        text.append(value)
        spans.append(Span(start=start, end=start + len(value), label=Label(label)))
        pos = m.end()
    text.append(raw[pos:])
    joined = "".join(text)
    if "[" in joined or "|" in joined:
        raise ValueError(f"{ex_id}: unparsed markup in {joined!r}")
    return Example(id=ex_id, text=joined, spans=spans, source="llm")


def load_drafts(seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for path in sorted(DRAFTS.glob("*.txt")):
        chunks = [c.strip() for c in path.read_text(encoding="utf-8").split("\n---\n")]
        for i, chunk in enumerate(c for c in chunks if c):
            examples.append(parse(chunk, f"{path.stem}-{i:03d}", rng))
    return examples


def _ls_task(ex: Example, with_predictions: bool) -> dict:
    task = {"data": {"id": ex.id, "text": ex.text}}
    if with_predictions:
        task["predictions"] = [{"model_version": "draft", "result": [
            {"id": f"{ex.id}-{k}", "from_name": "label", "to_name": "text", "type": "labels",
             "value": {"start": s.start, "end": s.end, "text": ex.text[s.start:s.end],
                       "labels": [s.label.value]}}
            for k, s in enumerate(ex.spans)]}]
    return task


def build() -> None:
    examples = load_drafts()
    blind = set(random.Random(42).sample([e.id for e in examples], BLIND_SIZE))
    with (OUT / "preannotated.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump(mode="json"), ensure_ascii=False) + "\n")
    # Слепые примеры идут без предразметки: по ним считаем согласие разметчиков.
    tasks = [_ls_task(e, e.id not in blind) for e in examples]
    (OUT / "label_studio_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "blind_ids.json").write_text(json.dumps(sorted(blind)), encoding="utf-8")
    labels = "\n".join(f'    <Label value="{l.value}"/>' for l in Label)
    (OUT / "label_studio_config.xml").write_text(
        f'<View>\n  <Labels name="label" toName="text">\n{labels}\n  </Labels>\n'
        f'  <Text name="text" value="$text"/>\n</View>\n', encoding="utf-8")
    n_spans = sum(len(e.spans) for e in examples)
    print(f"{len(examples)} examples, {n_spans} spans, {len(blind)} blind -> {OUT}")


def import_ls(export_path: Path) -> None:
    """Label Studio JSON export -> data/test/test.jsonl (source of truth for eval)."""
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    with (OUT / "test.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            ann = t["annotations"][0]["result"]
            spans = [Span(start=r["value"]["start"], end=r["value"]["end"], label=r["value"]["labels"][0])
                     for r in ann if r["type"] == "labels"]
            ex = Example(id=t["data"]["id"], text=t["data"]["text"], spans=spans, source="llm")
            f.write(json.dumps(ex.model_dump(mode="json"), ensure_ascii=False) + "\n")
    print(f"{len(tasks)} examples -> {OUT / 'test.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    imp = sub.add_parser("import")
    imp.add_argument("export", type=Path)
    args = ap.parse_args()
    build() if args.cmd == "build" else import_ls(args.export)


if __name__ == "__main__":
    main()
