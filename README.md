# Course Recommender

Hybrid recommender in Python: TF-IDF content similarity + item-based collaborative filtering.

## Setup
```bash
pip install -r requirements.txt
python sample_data.py          # creates data/courses.csv and data/ratings.csv (synthetic)
```

## Use
```bash
python recommender.py --user 3            # recommendations for a user
python recommender.py --course 1          # courses similar to a course
python recommender.py --evaluate          # leave-one-out hit-rate@5
python recommender.py --user 3 --alpha 0.8  # lean more on content similarity
```

```python
from recommender import CourseRecommender, load
rec = CourseRecommender(*load(), alpha=0.5)
rec.recommend(user_id=3, n=5)
```

## Using your own data
- `data/courses.csv`: `course_id, title, category, description, level`
- `data/ratings.csv`: `user_id, course_id, rating` (1-5)

## How it works
- **Content**: TF-IDF over title/category/level/description, cosine similarity between courses.
- **Collaborative**: user-mean-centered rating matrix, cosine similarity between courses.
- **Blend**: each course is scored by its similarity to courses the user rated (weighted by rating minus 3), min-max normalized, then mixed with `alpha`.
- **Cold start**: unknown users get the most popular courses.

## Web app (for end users)
```bash
pip install -r requirements.txt
python sample_data.py        # once, creates data/
streamlit run app.py         # opens http://localhost:8501
```
Users sign in by name, rate courses with stars, and get personalised picks (with reasons, filters and a variety
slider) that refresh as they rate. New users can type interests instead. Ratings persist in `app.db` (SQLite,
seeded from `data/ratings.csv`; delete the file to reset). To use your own catalog, replace the CSVs in `data/`
and delete `app.db`.
