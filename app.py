import pandas as pd
import streamlit as st

import storage
from recsys import HybridRecommender, load

st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")
storage.init()
courses, _ = load()
DESC = courses.set_index("course_id").description
LEVELS = ["beginner", "intermediate", "advanced"]


@st.cache_resource
def get_rec(version):
    return HybridRecommender().fit(courses, storage.load_ratings())


rec = get_rec(storage.version())

# ---------- sign in ----------
with st.sidebar:
    st.title("🎓 Course Recommender")
    if "uid" not in st.session_state:
        name = st.text_input("Your name", placeholder="e.g. Sara")
        if st.button("Get started", type="primary", disabled=not name.strip()):
            st.session_state.update(uid=storage.get_or_create_user(name), name=name.strip().title())
            st.rerun()
        with st.expander("Or try a demo profile"):
            demo = st.selectbox("Sample learner", sorted(rec.ratings.user_id.unique())[:10])
            if st.button("Use demo profile"):
                st.session_state.update(uid=int(demo), name=f"Demo learner {demo}")
                st.rerun()
    else:
        st.write(f"Signed in as **{st.session_state.name}**")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()

if "uid" not in st.session_state:
    st.header("Find your next course")
    st.write("Enter your name in the sidebar to get started. Rate a few courses and your recommendations improve.")
    st.stop()

uid = st.session_state.uid
mine = rec.user_ratings(uid)


def on_rate(cid, key):
    if st.session_state.get(key) is not None:
        storage.save_rating(uid, cid, st.session_state[key] + 1)


def card(row, key, reason=None):
    cid = int(row["course_id"])
    with st.container(border=True):
        left, right = st.columns([4, 2])
        left.markdown(f"**{row['title']}**  \n`{row['category']}` · `{row['level']}`")
        if reason:
            left.caption(reason)
        r = mine.get(cid)
        right.caption(f"Your rating: {'★' * int(r)}" if pd.notna(r) else "Rate this course")
        right.feedback("stars", key=key, on_change=on_rate, args=(cid, key))
        with st.expander("Details & similar courses"):
            st.write(DESC[cid])
            st.caption("More like this: " + ", ".join(rec.similar_courses(cid, 3).title))


tab_rec, tab_browse, tab_mine = st.tabs(["✨ For you", "📚 Browse & rate", "⭐ My ratings"])

with tab_rec:
    a, b, c = st.columns(3)
    levels = a.multiselect("Level", LEVELS) or None
    cats = b.multiselect("Category", sorted(courses.category.unique())) or None
    variety = c.slider("Variety", 0.0, 1.0, 0.2, help="Higher = a more varied mix of topics")
    if len(mine) < 3:
        st.info("Tell us what you're into, or rate a few courses in **Browse & rate** for personalised picks.")
        q = st.text_input("Your interests", placeholder="e.g. python, machine learning")
        recs = rec.recommend_for_interests(q, 6, levels, cats) if q else rec.recommend(uid, 6, levels, cats)
    else:
        recs = rec.recommend(uid, 6, levels, cats, variety)
    if recs.empty:
        st.warning("No courses match those filters. Try loosening them.")
    for _, r in recs.iterrows():
        card(r, f"rec_{r['course_id']}", r["reason"])

with tab_browse:
    q = st.text_input("Search courses", placeholder="python, security, design...")
    hits = courses
    if q:
        text = courses.title + " " + courses.category + " " + courses.description
        hits = courses[text.str.contains(q, case=False, regex=False)]
    for _, r in hits.sort_values("title").iterrows():
        card(r, f"browse_{r['course_id']}")
    if hits.empty:
        st.warning("No courses found.")

with tab_mine:
    if mine.empty:
        st.write("You haven't rated anything yet.")
    else:
        t = courses.set_index("course_id").loc[mine.index, ["title", "category", "level"]]
        t["rating"] = mine.astype(int).map(lambda x: "★" * x)
        st.dataframe(t.sort_values("rating", ascending=False))
