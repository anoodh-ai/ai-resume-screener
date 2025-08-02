import os
import joblib

# Paths
MODEL_PATH = r"C:\Users\anoodhivi\2025-2026 projects\ai-resume-screener\models\resume_model.pkl"
VECTORIZER_PATH = r"C:\Users\anoodhivi\2025-2026 projects\ai-resume-screener\models\vectorizer.pkl"
RESUME_FOLDER = r"C:\Users\anoodhivi\2025-2026 projects\ai-resume-screener\data"

# Load model & vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def read_resume(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().lower()


def analyze_resumes():
    results = []
    resumes = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".txt")]

    for resume in resumes:
        path = os.path.join(RESUME_FOLDER, resume)
        text = read_resume(path)

        # Convert resume to vector using saved vectorizer
        y = vectorizer.transform([text])
        prediction = model.predict(y)[0]

        results.append((resume, prediction))

    # Sort: relevant (1) first
    results.sort(key=lambda x: x[1], reverse=True)

    print("Resume Predictions (1 = Relevant, 0 = Not Relevant):")
    for res, pred in results:
        print(f"{res}: {pred}")

if __name__ == "__main__":
    analyze_resumes()

