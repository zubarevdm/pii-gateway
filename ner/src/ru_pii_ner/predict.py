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
        word_ids = enc.word_ids(0)
        logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        tags = [self.id2tag[int(i)] for i in logits.argmax(-1)]

        # Метку слова задаёт его первый подтокен, спан покрывает слово целиком:
        # иначе модель порождает обрывки вроде «ыл» из середины слова.
        words: dict[int, list] = {}
        for k, w in enumerate(word_ids):
            if w is None:
                continue
            if w not in words:
                words[w] = [offsets[k][0], offsets[k][1], tags[k]]
            else:
                words[w][1] = offsets[k][1]
        ordered = [words[w] for w in sorted(words)]
        spans = tags_to_spans([(s, e) for s, e, _ in ordered], [t for _, _, t in ordered])
        return postprocess(text, spans)


_NO_SPACE_LABELS = {"NICK", "EMAIL"}


def postprocess(text: str, spans: list[Span]) -> list[Span]:
    """Привести предсказания к правилам разметки (docs/annotation.md).

    1. Соседние спаны одной метки без пробела между ними склеиваются: токенизатор
       режет «@dark_knight_1997» по «_», и модель помечает не все куски.
    2. NICK и EMAIL не содержат пробелов, поэтому расширяются до целого слова.
    3. Границы нормализуются той же функцией, что и разметка теста: без маркеров
       («ИНН», «тел.») и краевой пунктуации, с точкой инициала.
    """
    from .adjudicate import normalize

    merged: list[Span] = []
    for s in sorted(spans, key=lambda x: x.start):
        prev = merged[-1] if merged else None
        if prev and prev.label == s.label and not any(ch.isspace() for ch in text[prev.end:s.start]):
            merged[-1] = Span(start=prev.start, end=max(prev.end, s.end), label=s.label)
        else:
            merged.append(s)

    out = []
    for s in merged:
        start, end = s.start, s.end
        if s.label.value in _NO_SPACE_LABELS:
            while start > 0 and not text[start - 1].isspace():
                start -= 1
            while end < len(text) and not text[end].isspace():
                end += 1
        n = normalize(text, Span(start=start, end=end, label=s.label))
        if n and any(ch.isalnum() for ch in text[n.start:n.end]):
            if out and out[-1].end > n.start:  # после расширения спаны могли наложиться
                continue
            out.append(n)
    return out
