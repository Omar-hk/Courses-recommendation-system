import os
import sqlite3
from contextlib import contextmanager

import pandas as pd


def db_path():
    return os.environ.get("COURSE_DB", "app.db")


@contextmanager
def db():
    c = sqlite3.connect(db_path())
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init(ratings_csv="data/ratings.csv"):
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS users(user_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS ratings(user_id INTEGER, course_id INTEGER, rating INTEGER,"
                  " PRIMARY KEY(user_id, course_id))")
        if c.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 0 and os.path.exists(ratings_csv):
            pd.read_csv(ratings_csv).to_sql("ratings", c, if_exists="append", index=False)


def get_or_create_user(name):
    name = name.strip().lower()
    with db() as c:
        row = c.execute("SELECT user_id FROM users WHERE name=?", (name,)).fetchone()
        if row:
            return int(row[0])
        top = max(c.execute("SELECT COALESCE(MAX(user_id),0) FROM ratings").fetchone()[0],
                  c.execute("SELECT COALESCE(MAX(user_id),0) FROM users").fetchone()[0])
        uid = max(10000, top + 1)  # keeps new users clear of the seeded sample users
        c.execute("INSERT INTO users VALUES(?,?)", (uid, name))
        return uid


def save_rating(user_id, course_id, rating):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO ratings VALUES(?,?,?)", (int(user_id), int(course_id), int(rating)))


def load_ratings():
    with db() as c:
        return pd.read_sql("SELECT user_id, course_id, rating FROM ratings", c)


def version():
    """Changes whenever any rating changes (used to refresh the cached model)."""
    with db() as c:
        return tuple(c.execute("SELECT COUNT(*), COALESCE(SUM(rating),0) FROM ratings").fetchone())
