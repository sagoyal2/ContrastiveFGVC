"""Shape and pooling invariants that need no GPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.registry import stratified_val_split
from src.match import Matcher, normalize, score
from src.models.base import Taps, pool_tokens


def test_pool_respects_valid_len():
    """The §13 padding trap: a pad row must not enter the average, and ``final``
    must land on the last real token rather than on padding."""
    x = torch.zeros(2, 5, 3)
    x[0, :3] = 1.0      # 3 real tokens, then padding
    x[1, :5] = 2.0
    avg = pool_tokens(x, "avg", [3, 5])
    assert torch.allclose(avg[0], torch.ones(3))
    assert torch.allclose(avg[1], 2 * torch.ones(3))
    fin = pool_tokens(x, "final", [3, 5])
    assert torch.allclose(fin[0], torch.ones(3))


def test_taps_validate_catches_bad_span():
    t = torch.zeros(1, 4, 8)
    taps = Taps(t0=torch.zeros(1, 4, 2), t1=torch.zeros(1, 4, 8),
                t2=torch.zeros(1, 4, 8), t3=torch.zeros(1, 10, 8),
                img_span=[(0, 3)], seq_len=[10])
    with pytest.raises(AssertionError):
        taps.validate()


def test_stratified_val_split_covers_every_class():
    labels = np.repeat(np.arange(200), 30)
    tr, va = stratified_val_split(labels, 0.1, 0)
    assert len(np.unique(labels[va])) == 200
    assert set(tr).isdisjoint(set(va))
    assert len(tr) + len(va) == len(labels)


def test_normalize_and_cascade_rungs():
    assert normalize("The Black-footed Albatrosses.") == "black footed albatross"
    m = Matcher(["black footed albatross", "laysan albatross"], "cub", use_embeddings=False)
    assert m.match_one("a Black-footed Albatross").rung == "exact"
    assert m.match_one("This is a black footed albatross bird").rung == "substring"
    assert m.match_one("a fire hydrant").rung == "unmatched"


def test_substring_prefers_the_longer_class_name():
    m = Matcher(["albatross", "black footed albatross"], "cub", use_embeddings=False)
    r = m.match_one("it looks like a black footed albatross to me")
    assert r.rung == "substring" and r.pred == 1


def test_score_reports_three_numbers():
    m = Matcher(["a", "b"], "cub", use_embeddings=False)
    res = m.match(["a", "this is a b thing", "zzz"])
    s = score(res, [0, 1, 0])
    assert s["strict"] < s["lenient"]
    assert s["pct_unmatched"] > 0
