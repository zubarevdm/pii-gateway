"""BIO encoding must round-trip: char spans -> token tags -> char spans."""
import pytest

transformers = pytest.importorskip("transformers")

from ru_pii_ner.generate import generate
from ru_pii_ner.predict import tags_to_spans
from ru_pii_ner.train import BASE_MODEL, TAGS, encode


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(BASE_MODEL)


def test_round_trip(tok):
    for ex in generate(300, seed=5):
        enc = encode(ex, tok, max_len=512)
        offsets = tok(ex.text, return_offsets_mapping=True)["offset_mapping"]
        tags = [TAGS[i] if i != -100 else "O" for i in enc["labels"].tolist()]
        got = [(s.start, s.end, s.label) for s in tags_to_spans(offsets, tags)]
        want = [(s.start, s.end, s.label) for s in ex.spans]
        assert got == want, (ex.text, got, want)


def test_adjacent_entities_stay_separate():
    offsets = [(0, 0), (0, 4), (5, 9), (10, 14), (0, 0)]
    tags = ["O", "B-PER", "B-PER", "I-PER", "O"]
    assert [(s.start, s.end) for s in tags_to_spans(offsets, tags)] == [(0, 4), (5, 14)]
