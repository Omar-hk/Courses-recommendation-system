from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from recsys import HybridRecommender, load

state = {}


@asynccontextmanager
async def lifespan(app):
    courses, ratings = load()
    state["rec"] = HybridRecommender().fit(courses, ratings)
    yield


app = FastAPI(title="Course Recommender", version="2.0", lifespan=lifespan)


class RatingIn(BaseModel):
    user_id: int
    course_id: int
    rating: int = Field(ge=1, le=5)


class InterestsIn(BaseModel):
    interests: str = Field(min_length=2, examples=["python machine learning"])
    levels: Optional[List[str]] = None
    n: int = Field(5, ge=1, le=20)


def rec():
    return state["rec"]


def check_course(course_id):
    if course_id not in rec().idx:
        raise HTTPException(404, f"Unknown course_id {course_id}")


@app.get("/health")
def health():
    return {"status": "ok", "courses": len(rec().courses), "ratings": len(rec().ratings)}


@app.get("/courses")
def courses():
    return rec().courses.to_dict("records")


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = Query(5, ge=1, le=20), level: Optional[List[str]] = Query(None),
              category: Optional[List[str]] = Query(None), diversity: float = Query(0.0, ge=0, le=1)):
    df = rec().recommend(user_id, n, level, category, diversity)
    return {"user_id": user_id, "known_user": user_id in rec()._by_user, "recommendations": df.to_dict("records")}


@app.get("/similar/{course_id}")
def similar(course_id: int, n: int = Query(5, ge=1, le=20)):
    check_course(course_id)
    return rec().similar_courses(course_id, n).to_dict("records")


@app.post("/recommend/interests")
def interests(body: InterestsIn):
    return rec().recommend_for_interests(body.interests, body.n, body.levels).to_dict("records")


@app.post("/ratings", status_code=201)
def add_rating(body: RatingIn):
    check_course(body.course_id)
    rec().add_rating(body.user_id, body.course_id, body.rating)
    return {"status": "recorded"}
