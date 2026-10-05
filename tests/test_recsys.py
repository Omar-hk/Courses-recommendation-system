import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from recsys import HybridRecommender, benchmark, load, split_ratings  # noqa: E402

os.chdir(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="module")
def data():
    return load()


@pytest.fixture(scope="module")
def rec(data):
    return HybridRecommender().fit(*data)


def test_never_recommends_already_rated(rec):
    uid = rec.ratings.user_id.iloc[0]
    taken = set(rec.user_ratings(uid).index)
    assert not taken & set(rec.recommend(uid, 10).course_id)


def test_filters_respected(rec):
    out = rec.recommend(3, 10, levels=["beginner"], categories=["Web", "Business"])
    assert len(out) and set(out.level) == {"beginner"} and set(out.category) <= {"Web", "Business"}


def test_cold_start_user_gets_popular(rec):
    out = rec.recommend(999999, 5)
    assert len(out) == 5 and (out.reason == "Popular with learners").all()


def test_interests_match_topic(rec):
    out = rec.recommend_for_interests("neural networks deep learning", 3)
    assert "Deep Learning with PyTorch" in set(out.title)


def _intra_list_sim(rec, ids):
    import numpy as np
    ix = [rec.idx[c] for c in ids]
    sub = rec.div_sim[np.ix_(ix, ix)]
    return (sub.sum() - len(ix)) / (len(ix) * (len(ix) - 1))


def test_diversity_lowers_intra_list_similarity(rec):
    plain = _intra_list_sim(rec, rec.recommend(3, 6, diversity=0.0).course_id)
    diverse = _intra_list_sim(rec, rec.recommend(3, 6, diversity=0.8).course_id)
    assert diverse <= plain


def test_split_has_no_leakage(data):
    train, test = split_ratings(data[1])
    assert not set(zip(train.user_id, train.course_id)) & set(zip(test.user_id, test.course_id))


def test_hybrid_beats_popularity(data):
    b = benchmark(*data)
    assert b.loc["hybrid", "ndcg@5"] > b.loc["popularity", "ndcg@5"]


def test_api():
    import api
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.get("/recommend/3?n=3&level=beginner").json()
        assert len(r["recommendations"]) <= 3
        assert c.get("/similar/99999").status_code == 404
        assert c.post("/ratings", json={"user_id": 7777, "course_id": 1, "rating": 9}).status_code == 422
        assert c.post("/ratings", json={"user_id": 7777, "course_id": 1, "rating": 5}).status_code == 201
        assert c.get("/recommend/7777").json()["known_user"] is True
