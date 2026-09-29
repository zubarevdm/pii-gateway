"""Evaluate a PII detector on the adjudicated test set.

    python -m ru_pii_ner.evaluate gateway       # regex + Natasha из app/detectors
    python -m ru_pii_ner.evaluate gateway-regex # только regex-слой

Результат пишется в eval/baseline-<name>.json: с ним сравнивается любая новая
модель. Метрики:
- strict — совпадают границы и метка;
- overlap — метка совпадает, спаны пересекаются (границы прощаются);
- masking recall — доля эталонных ПД, которые детектор хоть как-то закрыл,
  независимо от метки: для шлюза важно, что данные не ушли наружу.
ORG_ID считается отдельно и в сводную метрику по ПД не входит (решение 9),
NICK в сводной метрике объединяется с PER (решение 5).
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from .schema import Example, Span

TEST = Path("data/test/test.jsonl")
OUT = Path("eval")
GATEWAY_ROOT = Path(__file__).resolve().parents[3]

# EntityType шлюза -> метка датасета. ORG (название организации) и IP в схеме нет.
_GATEWAY_MAP = {
    "PERSON": "PER", "PHONE": "PHONE", "EMAIL": "EMAIL", "INN": "INN", "SNILS": "SNILS",
    "PASSPORT": "PASSPORT", "CARD": "CARD", "ACCOUNT": "ACCOUNT", "OGRN": "ORG_ID",
    "CAR_PLATE": "DOC_ID", "DOB": "BIRTHDATE", "LOCATION": "ADDRESS",
}

Predictor = Callable[[str], list[Span]]


def gateway_predictor(with_ner: bool = True) -> Predictor:
    sys.path.insert(0, str(GATEWAY_ROOT))
    from app.detectors.pipeline import DetectionPipeline
    from app.detectors.regex_detector import RegexDetector

    detectors = [RegexDetector()]
    if with_ner:
        from app.detectors.natasha_detector import NatashaNERDetector
        ner = NatashaNERDetector()
        ner.warmup()
        detectors.append(ner)
    pipeline = DetectionPipeline(detectors)

    def predict(text: str) -> list[Span]:
        return [Span(start=e.start, end=e.end, label=_GATEWAY_MAP[e.type.value])
                for e in pipeline.detect(text) if e.type.value in _GATEWAY_MAP and e.end > e.start]
    return predict


def _merge(label: str) -> str:
    return "PER" if label == "NICK" else label


def _counts(gold: list[Example], pred: dict[str, list[Span]], mode: str) -> tuple[Counter, Counter, Counter]:
    tp, fp, fn = Counter(), Counter(), Counter()
    for ex in gold:
        g = [(s.start, s.end, s.label.value) for s in ex.spans]
        p = [(s.start, s.end, s.label.value) for s in pred[ex.id]]
        if mode == "strict":
            gs, ps = set(g), set(p)
            for x in gs & ps: tp[x[2]] += 1
            for x in ps - gs: fp[x[2]] += 1
            for x in gs - ps: fn[x[2]] += 1
            continue
        used = set()
        for gi in g:
            hit = next((j for j, pj in enumerate(p) if j not in used and pj[2] == gi[2]
                        and pj[0] < gi[1] and gi[0] < pj[1]), None)
            if hit is None:
                fn[gi[2]] += 1
            else:
                used.add(hit); tp[gi[2]] += 1
        for j, pj in enumerate(p):
            if j not in used: fp[pj[2]] += 1
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def _pii_f1(gold: list[Example], pred: dict[str, list[Span]], mode: str) -> float:
    tp, fp, fn = _counts(gold, pred, mode)
    keep = lambda c: sum(v for k, v in c.items() if k != "ORG_ID")
    return _prf(keep(tp), keep(fp), keep(fn))["f1"]


def _masking_recall(gold: list[Example], pred: dict[str, list[Span]]) -> float:
    total = covered = 0
    for ex in gold:
        for g in ex.spans:
            if g.label.value == "ORG_ID":
                continue
            total += 1
            covered += any(p.start < g.end and g.start < p.end for p in pred[ex.id])
    return covered / total


def _bootstrap(gold: list[Example], pred: dict, fn: Callable, n: int = 1000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    vals = sorted(fn([rng.choice(gold) for _ in gold], pred) for _ in range(n))
    return [round(vals[int(0.025 * n)], 3), round(vals[int(0.975 * n)], 3)]


def evaluate(name: str, predict: Predictor) -> dict:
    gold = [Example.model_validate_json(l) for l in TEST.open(encoding="utf-8")]
    t0 = time.perf_counter()
    raw = {ex.id: predict(ex.text) for ex in gold}
    ms = (time.perf_counter() - t0) * 1000 / len(gold)
    merge = lambda spans: [Span(start=s.start, end=s.end, label=_merge(s.label.value)) for s in spans]
    gold_m = [Example(id=e.id, text=e.text, spans=merge(e.spans), source=e.source) for e in gold]
    pred_m = {k: merge(v) for k, v in raw.items()}

    report = {"name": name, "n_examples": len(gold), "ms_per_example": round(ms, 1), "summary": {}, "per_label": {}}
    for mode in ("strict", "overlap"):
        f = lambda g, p, m=mode: _pii_f1(g, p, m)
        report["summary"][f"pii_f1_{mode}"] = f(gold_m, pred_m)
        report["summary"][f"pii_f1_{mode}_ci95"] = _bootstrap(gold_m, pred_m, f)
        tp, fp, fn = _counts(gold, raw, mode)  # по меткам без объединения NICK
        report["per_label"][mode] = {l: _prf(tp[l], fp[l], fn[l]) for l in sorted(set(tp) | set(fp) | set(fn))}
    report["summary"]["masking_recall"] = round(_masking_recall(gold, raw), 3)
    report["summary"]["masking_recall_ci95"] = _bootstrap(gold, raw, _masking_recall)
    report["summary"]["pii_false_positives_on_clean_examples"] = sum(
        1 for e in gold if not [s for s in e.spans if s.label.value != "ORG_ID"]
        for s in raw[e.id] if s.label.value != "ORG_ID")

    OUT.mkdir(exist_ok=True)
    (OUT / f"baseline-{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"predictions-{name}.jsonl").write_text("".join(
        json.dumps({"id": i, "spans": [s.model_dump(mode="json") for s in v]}, ensure_ascii=False) + "\n"
        for i, v in raw.items()), encoding="utf-8")
    return report


def print_report(r: dict) -> None:
    s = r["summary"]
    print(f"== {r['name']}: {r['n_examples']} examples, {r['ms_per_example']} ms/example")
    print(f"PII F1 strict  {s['pii_f1_strict']:.3f}  95% CI {s['pii_f1_strict_ci95']}")
    print(f"PII F1 overlap {s['pii_f1_overlap']:.3f}  95% CI {s['pii_f1_overlap_ci95']}")
    print(f"masking recall {s['masking_recall']:.3f}  95% CI {s['masking_recall_ci95']}")
    print(f"false positives on examples without PII: {s['pii_false_positives_on_clean_examples']}")
    print(f"\n{'label':10} {'P':>6} {'R':>6} {'F1':>6} {'tp':>4} {'fp':>4} {'fn':>4}   (overlap)")
    for l, m in r["per_label"]["overlap"].items():
        print(f"{l:10} {m['p']:6.2f} {m['r']:6.2f} {m['f1']:6.2f} {m['tp']:4} {m['fp']:4} {m['fn']:4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("detector", choices=["gateway", "gateway-regex"])
    args = ap.parse_args()
    predict = gateway_predictor(with_ner=args.detector == "gateway")
    print_report(evaluate(args.detector, predict))


if __name__ == "__main__":
    main()
