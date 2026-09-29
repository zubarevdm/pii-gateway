"""Inference: token BIO tags -> character spans."""
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .schema import Label, Span


def tags_to_spans(offsets: list[tuple[int, int]], tags: list[str]) -> list[Span]:
    spans, cur = [], None  # cur = [start, end, label]
    for (s, e), tag in zip(offsets, tags):
        if s == e:  # служебные токены
            continue
        if tag == "O":
            cur = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and cur and cur[2] == label:
            cur[1] = e  # продолжение сущности
        else:  # B- или I- без начала: открываем новую сущность
            cur = [s, e, label]
            spans.append(cur)
    return [Span(start=s, end=e, label=Label(l)) for s, e, l in spans]


class NerModel:
    def __init__(self, path: Path | str, device: str | None = None, max_len: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device).eval()
        self.id2tag = self.model.config.id2label
        self.max_len = max_len

    @torch.no_grad()
    def __call__(self, text: str) -> list[Span]:
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        return tags_to_spans(offsets, [self.id2tag[int(i)] for i in logits.argmax(-1)])
