import pandas as pd

from .hybrid import DEFAULT_WEIGHTS, HybridRecommender
from .evaluate import benchmark, evaluate, split_ratings, tune_weights


def load(data_dir="data"):
    return pd.read_csv(f"{data_dir}/courses.csv"), pd.read_csv(f"{data_dir}/ratings.csv")
