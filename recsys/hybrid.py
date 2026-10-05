import numpy as np
import pandas as pd

from .models import ContentModel, ItemCFModel, PopularityModel, SVDModel, minmax

DEFAULT_WEIGHTS = {"content": 0.25, "itemcf": 0.25, "svd": 0.25, "popularity": 0.25}  # chosen via tune_weights on validation data
COLS = ["course_id", "title", "category", "level", "score", "reason"]


class HybridRecommender:
    def __init__(self, weights=None, like_threshold=4, svd_components=3):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.like_threshold = like_threshold
        self.svd_components = svd_components  # data-dependent: tune it (see README)

    # ---- fitting --------------------------------------------------------
    def fit(self, courses, ratings):
        self.courses = courses.reset_index(drop=True)
        self.ratings = ratings.drop_duplicates(["user_id", "course_id"], keep="last").reset_index(drop=True)
        self.idx = {c: i for i, c in enumerate(self.courses.course_id)}
        self.models = {m.name: m.fit(self.courses, self.ratings)
                       for m in (ContentModel(), ItemCFModel(), SVDModel(self.svd_components), PopularityModel())}
        self.content_sim = self.models["content"].item_sim
        off = self.content_sim[~np.eye(len(self.courses), dtype=bool)]
        self.div_sim = np.clip((self.content_sim - off.min()) / max(off.max() - off.min(), 1e-9), 0, 1)
        sims = [self.models[k].item_sim for k in ("content", "itemcf", "svd")]
        self.item_sim = sum(sims) / len(sims)
        self._by_user = {u: g.set_index("course_id").rating for u, g in self.ratings.groupby("user_id")}
        return self

    def add_rating(self, user_id, course_id, rating):
        new = pd.DataFrame([(user_id, course_id, rating)], columns=["user_id", "course_id", "rating"])
        return self.fit(self.courses, pd.concat([self.ratings, new], ignore_index=True))

    def user_ratings(self, user_id):
        return self._by_user.get(user_id, pd.Series(dtype=float))

    # ---- core ranking ---------------------------------------------------
    def _mask(self, rated, levels, categories):
        mask = ~self.courses.course_id.isin(rated.index).values
        if levels:
            mask &= self.courses.level.isin(levels).values
        if categories:
            mask &= self.courses.category.isin(categories).values
        return mask

    def _blend(self, rated, mask, weights):
        total, wsum = np.full(len(self.courses), -np.inf), 0.0
        total[mask] = 0.0
        for name, w in weights.items():
            if w > 0:
                total[mask] += w * minmax(self.models[name].score(rated)[mask])
                wsum += w
        total[mask] /= max(wsum, 1e-9)
        return total

    def _mmr(self, score, mask, n, lam):
        """Maximal Marginal Relevance: trade relevance against similarity to already-picked items."""
        cand, chosen = list(np.where(mask)[0]), []
        while cand and len(chosen) < n:
            def mmr(i):
                red = max(self.div_sim[i, j] for j in chosen)
                return lam * score[i] - (1 - lam) * red
            best = max(cand, key=lambda i: score[i]) if not chosen else max(cand, key=mmr)
            chosen.append(best)
            cand.remove(best)
        return chosen

    def rank(self, rated, n=5, weights=None, levels=None, categories=None, diversity=0.0):
        mask = self._mask(rated, levels, categories)
        if not mask.any():
            return [], np.array([])
        score = self._blend(rated, mask, weights or self.weights)
        if diversity > 0:
            return self._mmr(score, mask, n, 1 - diversity), score
        return list(np.argsort(-score)[: min(n, mask.sum())]), score

    # ---- public API -----------------------------------------------------
    def recommend(self, user_id, n=5, levels=None, categories=None, diversity=0.0):
        rated = self.user_ratings(user_id)
        weights = self.weights if len(rated) else {"popularity": 1.0}  # cold start
        order, score = self.rank(rated, n, weights, levels, categories, diversity)
        return self._frame(order, score, [self._explain(i, rated) for i in order])

    def recommend_for_interests(self, text, n=5, levels=None, categories=None):
        """Cold start for brand-new users: free-text interests, e.g. 'python machine learning'."""
        mask = self._mask(pd.Series(dtype=float), levels, categories)
        q = minmax(self.models["content"].query_scores(text)[mask])
        p = minmax(self.models["popularity"].score(None)[mask])
        score = np.full(len(self.courses), -np.inf)
        score[mask] = 0.8 * q + 0.2 * p
        order = list(np.argsort(-score)[: min(n, mask.sum())])
        return self._frame(order, score, [f"Matches your interests: {text}"] * len(order))

    def similar_courses(self, course_id, n=5):
        i = self.idx[course_id]
        order = [j for j in np.argsort(-self.item_sim[i]) if j != i][:n]
        df = self.courses.iloc[order][["course_id", "title", "category", "level"]].copy()
        df["similarity"] = self.item_sim[i][order]
        return df

    # ---- helpers --------------------------------------------------------
    def _explain(self, i, rated):
        """Name the liked course that contributed most: (rating - 3) * similarity."""
        liked = rated[rated >= self.like_threshold]
        if liked.empty:
            return "Popular with learners"
        contrib = {c: (r - 3.0) * self.item_sim[i, self.idx[c]] for c, r in liked.items()}
        cid = max(contrib, key=contrib.get)
        if contrib[cid] <= 0.02:
            return "Highly rated overall"
        return f"Because you rated '{self.courses.title[self.idx[cid]]}' {int(liked[cid])}/5"

    def _frame(self, order, score, reasons):
        df = self.courses.iloc[order].copy()
        df["score"] = np.round(score[order], 4)
        df["reason"] = reasons
        return df[COLS].reset_index(drop=True)
