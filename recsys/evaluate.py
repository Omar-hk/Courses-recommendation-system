import itertools
import numpy as np
import pandas as pd

from .hybrid import DEFAULT_WEIGHTS, HybridRecommender


def split_ratings(ratings, test_frac=0.2, threshold=4, min_ratings=5, seed=0):
    """Hold out a fraction of each user's *liked* ratings as test positives."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, g in ratings.groupby("user_id"):
        pos = g[g.rating >= threshold]
        if len(g) >= min_ratings and len(pos) >= 2:
            k = max(1, round(test_frac * len(pos)))
            test_idx += list(rng.choice(pos.index, size=k, replace=False))
    return ratings.drop(test_idx), ratings.loc[test_idx]


def _ndcg(hits, n_rel, k):
    dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits[:k]) if h)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / idcg if idcg else 0.0


def evaluate(rec, train, test, weights=None, k=5, diversity=0.0):
    by_user = {u: g.set_index("course_id").rating for u, g in train.groupby("user_id")}
    p, r, n, h, seen = [], [], [], [], set()
    for uid, t in test.groupby("user_id"):
        if uid not in by_user:
            continue
        order, _ = rec.rank(by_user[uid], k, weights, diversity=diversity)
        ids = [rec.courses.course_id[i] for i in order]
        rel = set(t.course_id)
        hits = [c in rel for c in ids]
        p.append(sum(hits) / k)
        r.append(sum(hits) / len(rel))
        n.append(_ndcg(hits, len(rel), k))
        h.append(float(any(hits)))
        seen.update(ids)
    return {f"precision@{k}": np.mean(p), f"recall@{k}": np.mean(r), f"ndcg@{k}": np.mean(n),
            f"hit@{k}": np.mean(h), "coverage": len(seen) / len(rec.courses), "users": len(p)}


def benchmark(courses, ratings, k=5, seed=0, weights=None):
    train, test = split_ratings(ratings, seed=seed)
    rec = HybridRecommender(weights).fit(courses, train)
    configs = {"popularity": {"popularity": 1}, "content": {"content": 1},
               "itemcf": {"itemcf": 1}, "svd": {"svd": 1}, "hybrid": rec.weights}
    rows = {name: evaluate(rec, train, test, w, k) for name, w in configs.items()}
    rows["hybrid + MMR(0.3)"] = evaluate(rec, train, test, rec.weights, k, diversity=0.3)
    return pd.DataFrame(rows).T


def tune_weights(courses, ratings, k=5, step=0.25, seed=1):
    """Grid-search blend weights on a validation split carved out of the training data."""
    train, _ = split_ratings(ratings, seed=0)  # never touch the benchmark's test rows
    fit_part, val = split_ratings(train, seed=seed)
    rec = HybridRecommender().fit(courses, fit_part)
    names, grid, results = list(DEFAULT_WEIGHTS), np.arange(0, 1 + 1e-9, step), []
    for combo in itertools.product(grid, repeat=len(names)):
        if abs(sum(combo) - 1) > 1e-9:
            continue
        w = dict(zip(names, combo))
        results.append({**w, **{"ndcg@%d" % k: evaluate(rec, fit_part, val, w, k)["ndcg@%d" % k]}})
    return pd.DataFrame(results).sort_values(f"ndcg@{k}", ascending=False).reset_index(drop=True)
