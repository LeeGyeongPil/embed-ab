"""Okapi BM25 in numpy, matching ``rank_bm25.BM25Okapi`` scores (k1=1.5, b=0.75, epsilon=0.25)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

TOKENIZERS = ("word", "char2")


def tokenize(text: str, kind: str = "word") -> list[str]:
    """``word``: lower-cased ``\\w+`` tokens. ``char2``: character bigrams inside each word.

    Character bigrams make Korean particles ("백업을" / "백업은") share the stem bigram "백업".
    """
    words = re.findall(r"\w+", text.lower())
    if kind == "word":
        return words
    if kind != "char2":
        raise ValueError(f"Unknown BM25 tokenizer {kind!r}; choose one of {', '.join(TOKENIZERS)}.")
    out: list[str] = []
    for w in words:
        out.extend([w] if len(w) < 2 else [w[i : i + 2] for i in range(len(w) - 1)])
    return out


class BM25:
    def __init__(self, documents: Sequence[str], tokenizer: str = "word", k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.tokenizer = tokenizer
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d, tokenizer)) for d in documents]
        self.lengths = np.array([sum(c.values()) for c in self.docs], dtype=np.float64)
        n = len(self.docs)
        self.avgdl = float(self.lengths.mean()) if n else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        idf = {t: math.log((n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        floor = epsilon * (sum(idf.values()) / len(idf) if idf else 0.0)
        self.idf = {t: (v if v >= 0 else floor) for t, v in idf.items()}  # rank_bm25 floors negative idf

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1.0))
        for term in tokenize(query, self.tokenizer):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self.docs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
