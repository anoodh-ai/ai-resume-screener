import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample training data (later can expand)
data = {
    "text": [
        # Relevant resumes (1)
        "python machine learning sql azure javascript",
        "azure cloud devops python sql pipelines",
        "data science python sql machine learning azure",
        "python flask django azure cloud deployment",
        "machine learning deep learning python sql",
        "python sql javascript azure data engineering",
        "javascript react node azure python sql",
        "cloud engineer python azure devops pipelines",
        "python sql tableau data analysis azure",
        "azure ml python machine learning devops",
        "python data pipelines sql azure cloud",
        "machine learning python sql javascript react",
        "python django flask azure devops",
        "azure kubernetes python cloud engineer",
        "python sql data analysis machine learning",
        "python tensorflow keras azure deep learning",
        "python javascript sql azure pipelines",
        "python cloud functions azure devops",
        "python ai ml sql azure automation",
        "machine learning python sql cloud pipelines",
        "azure python sql kubernetes ml",
        "python devops azure pipelines ci cd",
        "python sql ai ml cloud projects",
        "python flask azure ml deployment",
        "python azure sql data engineering",

        # Not relevant resumes (0)
        "basic html css javascript design",
        "c programming beginner without cloud",
        "photoshop illustrator graphic design",
        "excel powerpoint word office tools",
        "networking fundamentals hardware support",
        "basic java programming oops concepts",
        "graphic design adobe photoshop illustrator",
        "core electronics microcontroller pcb design",
        "network security cisco ccn a beginner",
        "basic computer skills ms office word",
        "graphic design photoshop illustrator beginner",
        "electrical engineering circuits motors basics",
        "basic accounting tally excel bookkeeping",
        "network administrator beginner hardware support",
        "basic video editing filmora premiere pro",
        "non technical support bpo voice process",
        "basic typing ms word excel ppt",
        "sales executive marketing beginner fresher",
        "content writing basic seo wordpress",
        "basic teaching tutor kindergarten experience",
        "hotel management catering kitchen staff",
        "basic store management retail operations",
        "beginner call center voice support",
        "basic computer operator office assistant"
    ],
    "label": [
        # 25 relevant (1)
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        # 25 not relevant (0)
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]
}


df = pd.DataFrame(data)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
# Save model & vectorizer
joblib.dump(model, "../models/resume_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")