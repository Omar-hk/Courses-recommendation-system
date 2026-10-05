"""Individual scoring models. Every model exposes:
    fit(courses, ratings) -> self
    score(rated: pd.Series[course_id -> rating]) -> np.ndarray over courses (higher = better)
    item_sim: (n_courses x n_courses) similarity matrix
Scoring from a `rated` Series (not a user id) means any model can score unseen/held-out users.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEUTRAL = 3.0


def minmax(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 1e-12 else np.zeros_like(x)


def rating_matrix(courses, ratings):
    m = ratings.pivot_table(index="user_id", columns="course_id", values="rating")
    return m.reindex(columns=courses.course_id)


def sim_score(sim, rated, idx):
    """Rating-weighted similarity to rated courses (liked pulls up, disliked pushes down)."""
    pos = [idx[c] for c in rated.index if c in idx]
    w = rated[[c for c in rated.index if c in idx]].values - NEUTRAL
    return w @ sim[pos] if pos else np.zeros(sim.shape[0])


class ContentModel:
    """TF-IDF over course text; similarity between courses."""
    name = "content"

    def fit(self, courses, ratings):
        self.idx = {c: i for i, c in enumerate(courses.course_id)}
        text = courses.title + " " + courses.category + " " + courses.level + " " + courses.description
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.tfidf.fit_transform(text)
        self.item_sim = cosine_similarity(self.X)
        return self

    def score(self, rated):
        return sim_score(self.item_sim, rated, self.idx)

    def query_scores(self, text):
        """Cold start: match free-text interests against the catalog."""
        return cosine_similarity(self.tfidf.transform([text]), self.X)[0]


class ItemCFModel:
    """Item-based collaborative filtering with per-user centering and co-rating shrinkage."""
    name = "itemcf"

    def __init__(self, shrink=10):
        self.shrink = shrink

    def fit(self, courses, ratings):
        self.idx = {c: i for i, c in enumerate(courses.course_id)}
        m = rating_matrix(courses, ratings)
        centered = m.sub(m.mean(axis=1), axis=0).fillna(0).values
        seen = m.notna().astype(float).values
        co = seen.T @ seen  # users who rated both items
        self.item_sim = cosine_similarity(centered.T) * (co / (co + self.shrink))
        return self

    def score(self, rated):
        return sim_score(self.item_sim, rated, self.idx)


class SVDModel:
    """Latent-factor model (truncated SVD). New users are 'folded in' from their ratings."""
    name = "svd"

    def __init__(self, n_components=8, seed=0):
        self.n_components, self.seed = n_components, seed

    def fit(self, courses, ratings):
        self.idx = {c: i for i, c in enumerate(courses.course_id)}
        m = rating_matrix(courses, ratings)
        centered = m.sub(m.mean(axis=1), axis=0).fillna(0).values
        k = max(1, min(self.n_components, centered.shape[0] - 1, centered.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(centered)
        self.item_sim = cosine_similarity(self.svd.components_.T)
        return self

    def score(self, rated):
        vec = np.zeros(len(self.idx))
        for c, r in (rated - rated.mean()).items():
            if c in self.idx:
                vec[self.idx[c]] = r
        return self.svd.inverse_transform(self.svd.transform(vec[None]))[0]


class PopularityModel:
    """Bayesian-smoothed average rating (baseline + cold-start prior)."""
    name = "popularity"

    def __init__(self, m=5):
        self.m = m

    def fit(self, courses, ratings):
        g = ratings.groupby("course_id").rating.agg(["mean", "count"])
        glob = ratings.rating.mean()
        g["bayes"] = (g["count"] * g["mean"] + self.m * glob) / (g["count"] + self.m)
        self.pop = g["bayes"].reindex(courses.course_id).fillna(glob).values
        self.item_sim = np.zeros((len(courses), len(courses)))
        return self

    def score(self, rated):
        return self.pop
