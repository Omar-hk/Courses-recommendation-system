import os
import numpy as np
import pandas as pd

COURSES = [
    ("Python for Beginners", "Programming", "Learn Python syntax variables loops functions and basic scripting", "beginner"),
    ("Advanced Python", "Programming", "Decorators generators async concurrency and performance in Python", "advanced"),
    ("Data Structures and Algorithms", "Programming", "Arrays trees graphs sorting searching and complexity analysis", "intermediate"),
    ("Web Development with Django", "Web", "Build web applications with Python Django models views templates and REST APIs", "intermediate"),
    ("HTML CSS and JavaScript Basics", "Web", "Create responsive websites with HTML CSS and JavaScript fundamentals", "beginner"),
    ("React Front-End Development", "Web", "Build interactive user interfaces with React components hooks and state", "intermediate"),
    ("Intro to Machine Learning", "Data Science", "Supervised learning regression classification and model evaluation with Python", "intermediate"),
    ("Deep Learning with PyTorch", "Data Science", "Neural networks convolutional recurrent models and training with PyTorch", "advanced"),
    ("Data Analysis with Pandas", "Data Science", "Clean transform and analyze data with Python pandas and visualization", "beginner"),
    ("Statistics for Data Science", "Data Science", "Probability distributions hypothesis testing and regression analysis", "intermediate"),
    ("SQL and Databases", "Data", "Relational databases SQL queries joins indexing and schema design", "beginner"),
    ("Big Data with Spark", "Data", "Distributed data processing with Apache Spark and large datasets", "advanced"),
    ("Cloud Computing Fundamentals", "Cloud", "Cloud services virtual machines storage networking and deployment basics", "beginner"),
    ("DevOps with Docker and Kubernetes", "Cloud", "Containers orchestration CI/CD pipelines and infrastructure automation", "intermediate"),
    ("Cybersecurity Essentials", "Security", "Threats encryption network security and secure coding practices", "beginner"),
    ("Ethical Hacking", "Security", "Penetration testing vulnerability scanning and exploit analysis", "advanced"),
    ("UX Design Principles", "Design", "User research wireframing prototyping and usability testing", "beginner"),
    ("Digital Marketing Basics", "Business", "SEO social media advertising and analytics for online growth", "beginner"),
    ("Project Management", "Business", "Agile scrum planning risk management and team leadership", "intermediate"),
    ("Natural Language Processing", "Data Science", "Text processing embeddings transformers and language models with Python", "advanced"),
]

# Users lean toward a "taste" cluster so collaborative filtering has signal.
TASTES = {
    "Programming": ["Programming", "Web", "Cloud"],
    "DataSci": ["Data Science", "Data"],
    "Security": ["Security", "Cloud", "Programming"],
    "Business": ["Business", "Design", "Web"],
}

def main(n_users=200, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs("data", exist_ok=True)
    courses = pd.DataFrame(COURSES, columns=["title", "category", "description", "level"])
    courses.insert(0, "course_id", range(1, len(courses) + 1))
    courses.to_csv("data/courses.csv", index=False)

    rows = []
    tastes = list(TASTES)
    for uid in range(1, n_users + 1):
        liked = TASTES[tastes[uid % len(tastes)]]
        taken = rng.choice(courses.course_id, size=rng.integers(5, 10), replace=False)
        for cid in taken:
            cat = courses.loc[courses.course_id == cid, "category"].iloc[0]
            base = 4.3 if cat in liked else 2.4
            rows.append((uid, int(cid), int(np.clip(round(rng.normal(base, 0.7)), 1, 5))))
    pd.DataFrame(rows, columns=["user_id", "course_id", "rating"]).to_csv("data/ratings.csv", index=False)
    print(f"Wrote {len(courses)} courses and {len(rows)} ratings to ./data")

if __name__ == "__main__":
    main()
