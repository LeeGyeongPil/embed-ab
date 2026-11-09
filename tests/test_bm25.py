import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from embed_ab.bm25 import BM25, tokenize

DOCS = [
    "the cat sat on the mat",
    "dogs chase the cat around the garden",
    "a quick brown fox jumps over the lazy dog",
    "cats and dogs and foxes",
    "nothing relevant here at all",
]


def test_matches_rank_bm25():
    ours = BM25(DOCS, "word")
    ref = BM25Okapi([tokenize(d) for d in DOCS])
    for query in ["cat", "the cat dog", "fox jumps", "unknownword", "the"]:
        expected = ref.get_scores(tokenize(query))
        assert np.allclose(ours.scores(query), expected, atol=1e-9), query


def test_char2_handles_korean_particles():
    docs = ["백업은 설정에서 바꿉니다", "동기화는 와이파이에서만 됩니다", "암호는 복구할 수 없습니다", "공유 링크는 만료됩니다"]
    query = "백업을 어떻게 바꾸나요"  # "백업을" != "백업은" as words
    word = BM25(docs, "word").scores(query)
    char2 = BM25(docs, "char2").scores(query)
    assert word[0] == pytest.approx(0.0) or word[0] <= word[1]
    assert char2[0] > char2[1]


def test_unknown_tokenizer():
    with pytest.raises(ValueError, match="Unknown BM25 tokenizer"):
        tokenize("x", "bogus")
