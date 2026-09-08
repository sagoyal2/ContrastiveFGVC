"""The generative matching cascade (§6.1).

Open-ended generation has to be turned into a class prediction, and *how* you do
that is the single biggest source of irreproducibility in this literature. So the
cascade is explicit, ordered, and every prediction records which rung fired:

1. normalise      lowercase, strip articles/punctuation, collapse whitespace, singularise
2. exact          normalised generation == normalised class name
3. alias          exact against a hand-maintained alias table
4. substring      class name appears inside the generation
5. embedding      nearest class by sentence embedding, cosine >= 0.65
6. unmatched

Rungs 1-3 give the **strict** number, 1-5 the **lenient** number, and the
remainder is **%unmatched**. Never report just one of the three.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .paths import CONFIGS_DIR

ARTICLES = ("a ", "an ", "the ")
TAU = 0.65
RUNGS = ("exact", "alias", "substring", "embedding", "unmatched")
STRICT_RUNGS = ("exact", "alias")
LENIENT_RUNGS = ("exact", "alias", "substring", "embedding")

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    for a in ARTICLES:
        if s.startswith(a):
            s = s[len(a):]
    return _singularize(s)


def _singularize(s: str) -> str:
    """Crude, deliberately. A real lemmatiser would fold 'Cardinals'->'Cardinal'
    but also mangle Latinate class names ('albatross' -> 'albatros'), so only the
    unambiguous plural endings are touched."""
    words = s.split()
    if not words:
        return s
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ses") or w.endswith("shes") or w.endswith("ches"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    words[-1] = w
    return " ".join(words)


def load_aliases(dataset: str) -> Dict[str, str]:
    """``aliases.json``: alias -> canonical class name, both normalised.

    ImageNet's multi-synonym labels are split on ',' automatically; anything else
    is hand-maintained per dataset.
    """
    path = CONFIGS_DIR / "aliases.json"
    if not path.exists():
        return {}
    table = json.loads(path.read_text())
    return {normalize(k): normalize(v) for k, v in table.get(dataset, {}).items()}


@dataclass
class MatchResult:
    pred: Optional[int]     # class index, or None if unmatched
    rung: str
    score: float = 1.0


class Matcher:
    def __init__(self, classnames: Sequence[str], dataset: str, use_embeddings: bool = True):
        self.classnames = list(classnames)
        self.norm_names = [normalize(c) for c in self.classnames]
        self.by_name = {n: i for i, n in enumerate(self.norm_names)}

        self.aliases: Dict[str, int] = {}
        for alias, canon in load_aliases(dataset).items():
            if canon in self.by_name:
                self.aliases[alias] = self.by_name[canon]
        # ImageNet-style "tench, Tinca tinca" labels: every comma-separated
        # synonym is an alias for the same class, for free.
        for i, c in enumerate(self.classnames):
            if "," in c:
                for part in c.split(","):
                    self.aliases.setdefault(normalize(part), i)

        self._embedder = None
        self._class_emb = None
        if use_embeddings:
            self._try_load_embedder()

    def _try_load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            # Rung 5 is the optional `match` extra (pyproject). Without it the
            # cascade stops at rung 4 and reports a higher %unmatched rather than
            # failing — an honest degradation, not a silent one.
            print("[match] sentence-transformers absent; cascade stops at rung 4")
            return
        self._embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self._class_emb = self._embedder.encode(
            self.classnames, normalize_embeddings=True, show_progress_bar=False
        )

    def match_one(self, generation: str) -> MatchResult:
        g = normalize(generation)
        if not g:
            return MatchResult(None, "unmatched", 0.0)
        if g in self.by_name:
            return MatchResult(self.by_name[g], "exact")
        if g in self.aliases:
            return MatchResult(self.aliases[g], "alias")
        # Longest containing class name wins: "black footed albatross" must beat
        # "albatross" when both appear in the generation.
        hits = [(len(n), i) for n, i in self.by_name.items() if n and n in g]
        if hits:
            return MatchResult(max(hits)[1], "substring")
        return MatchResult(None, "unmatched", 0.0)

    def match(self, generations: Sequence[str]) -> List[MatchResult]:
        out = [self.match_one(g) for g in generations]
        if self._embedder is None:
            return out

        todo = [i for i, r in enumerate(out) if r.rung == "unmatched"]
        if not todo:
            return out
        emb = self._embedder.encode(
            [generations[i] for i in todo], normalize_embeddings=True, show_progress_bar=False
        )
        sims = emb @ self._class_emb.T
        for k, i in enumerate(todo):
            j = int(sims[k].argmax())
            s = float(sims[k, j])
            if s >= TAU:
                out[i] = MatchResult(j, "embedding", s)
            else:
                out[i] = MatchResult(None, "unmatched", s)
        return out


def score(results: Sequence[MatchResult], labels: Sequence[int]) -> Dict[str, float]:
    """strict / lenient / %unmatched, plus the per-rung histogram (§6.1)."""
    n = len(results)
    strict = sum(r.rung in STRICT_RUNGS and r.pred == y for r, y in zip(results, labels))
    lenient = sum(r.rung in LENIENT_RUNGS and r.pred == y for r, y in zip(results, labels))
    unmatched = sum(r.rung == "unmatched" for r in results)
    hist = {rung: sum(r.rung == rung for r in results) for rung in RUNGS}
    return {
        "strict": 100 * strict / n,
        "lenient": 100 * lenient / n,
        "pct_unmatched": 100 * unmatched / n,
        "n": n,
        "rungs": hist,
    }
