import os
import re

# Path to resumes
RESUME_FOLDER = r"C:\Users\anoodhivi\2025-2026 projects\ai-resume-screener\data"

# Define keywords (skills required for job)
KEYWORDS = ["python", "machine learning", "sql", "azure", "javascript"]


def read_resume(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().lower()

def calculate_score(text):
    score = 0
    for word in KEYWORDS:
        if re.search(rf"\b{word}\b", text):
            score += 1
    return (score / len(KEYWORDS)) * 100  # percentage

def analyze_resumes():
    results = []
    resumes = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".txt")]

    for resume in resumes:
        path = os.path.join(RESUME_FOLDER, resume)
        text = read_resume(path)
        score = calculate_score(text)
        results.append((resume, score))

    # Sort high to low score
    results.sort(key=lambda x: x[1], reverse=True)

    print("Resume Rankings:")
    for res, score in results:
        print(f"{res}: {score:.2f}% match")

if __name__ == "__main__":
    analyze_resumes()