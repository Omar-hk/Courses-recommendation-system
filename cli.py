import argparse
import pandas as pd

from recsys import HybridRecommender, benchmark, load, tune_weights

pd.set_option("display.width", 160, "display.max_colwidth", 60)


def csv_list(s):
    return [x.strip() for x in s.split(",")] if s else None


def main():
    p = argparse.ArgumentParser(description="Course recommender")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("recommend", help="recommendations for a user")
    r.add_argument("user", type=int)
    r.add_argument("-n", type=int, default=5)
    r.add_argument("--level", type=csv_list, help="e.g. beginner,intermediate")
    r.add_argument("--category", type=csv_list, help="e.g. 'Data Science,Web'")
    r.add_argument("--diversity", type=float, default=0.0, help="0-1, MMR re-ranking")

    s = sub.add_parser("similar", help="courses similar to a course")
    s.add_argument("course", type=int)
    s.add_argument("-n", type=int, default=5)

    i = sub.add_parser("interests", help="cold start from free-text interests")
    i.add_argument("text")
    i.add_argument("-n", type=int, default=5)
    i.add_argument("--level", type=csv_list)

    sub.add_parser("benchmark", help="compare all models on a held-out test set")
    t = sub.add_parser("tune", help="grid-search blend weights")
    t.add_argument("--top", type=int, default=5)

    a = p.parse_args()
    courses, ratings = load()

    if a.cmd == "benchmark":
        print(benchmark(courses, ratings).round(3).to_string())
    elif a.cmd == "tune":
        print(tune_weights(courses, ratings).head(a.top).round(3).to_string(index=False))
    else:
        rec = HybridRecommender().fit(courses, ratings)
        if a.cmd == "recommend":
            out = rec.recommend(a.user, a.n, a.level, a.category, a.diversity)
        elif a.cmd == "similar":
            out = rec.similar_courses(a.course, a.n)
        else:
            out = rec.recommend_for_interests(a.text, a.n, a.level)
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
