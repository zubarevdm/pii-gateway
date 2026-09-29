"""Fine-tune rubert-tiny2 for PII token classification.

    python -m ru_pii_ner.train --out models/rubert-tiny2-pii

Разметка по символам переводится в BIO-теги по токенам через offset_mapping:
первый токен сущности получает B-, остальные I-. Тестовый набор здесь не
используется: отложенная выборка для контроля берётся из той же синтетики,
тест остаётся только для финального замера.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .schema import Example, Label

BASE_MODEL = "cointegrated/rubert-tiny2"
TAGS = ["O"] + [f"{p}-{l.value}" for l in Label for p in "BI"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}


def encode(ex: Example, tok, max_len: int) -> dict:
    enc = tok(ex.text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    labels, started = [], set()
    for s, e in enc["offset_mapping"]:
        if s == e:  # [CLS], [SEP]
            labels.append(-100)
            continue
        tag = "O"
        for k, sp in enumerate(ex.spans):
            if s < sp.end and e > sp.start:
                tag = ("I-" if k in started else "B-") + sp.label.value
                started.add(k)
                break
        labels.append(TAG2ID[tag])
    return {"input_ids": torch.tensor(enc["input_ids"]), "labels": torch.tensor(labels)}


def collate(batch: list[dict], pad_id: int) -> dict:
    ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    return {"input_ids": ids, "attention_mask": (ids != pad_id).long(), "labels": labels}


def load(path: Path) -> list[Example]:
    return [Example.model_validate_json(l) for l in path.open(encoding="utf-8")]


@torch.no_grad()
def token_accuracy(model, loader, device) -> float:
    """Доля верно предсказанных не-O токенов: быстрый сигнал, что обучение идёт."""
    model.eval()
    hit = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.argmax(-1)
        mask = (batch["labels"] != -100) & (batch["labels"] != TAG2ID["O"])
        hit += (pred[mask] == batch["labels"][mask]).sum().item()
        total += mask.sum().item()
    model.train()
    return hit / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/synthetic/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("models/rubert-tiny2-pii"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ids = {json.loads(l)["id"] for l in Path("data/test/test.jsonl").open(encoding="utf-8")}
    examples = load(args.train)
    assert not test_ids & {e.id for e in examples}, "примеры теста попали в обучение"
    random.shuffle(examples)
    n_dev = max(len(examples) // 10, 1)
    dev, train = examples[:n_dev], examples[n_dev:]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id=TAG2ID).to(device)

    make = lambda data, shuffle: DataLoader(
        [encode(e, tok, args.max_len) for e in data], batch_size=args.batch, shuffle=shuffle,
        collate_fn=lambda b: collate(b, tok.pad_token_id))
    train_dl, dev_dl = make(train, True), make(dev, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    print(f"device={device} train={len(train)} dev={len(dev)} steps={steps}")
    t0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch}: loss {total / len(train_dl):.4f}  dev entity-token acc "
              f"{token_accuracy(model, dev_dl, device):.4f}  ({time.time() - t0:.0f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "train_args.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()} | {"base_model": BASE_MODEL, "n_train": len(train)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
