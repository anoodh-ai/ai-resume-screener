import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample training data (later can expand)
data = {
    "text": [
        # Relevant (1)
        "python machine learning sql azure",
        "javascript azure react cloud devops",
        "sql python data analysis azure pipelines",
        "machine learning deep learning python",
        "python azure javascript sql cloud",
        "ai chatbot langchain javascript python",
        "data science pandas numpy sklearn azure",
        "azure devops cloud pipelines automation",
        "natural language processing nlp python",
        "sql data warehouse azure data engineer",
        "computer vision python opencv deep learning",
        "react typescript azure full stack developer",
        "azure functions python serverless backend",
        "predictive analytics sales forecast python",
        "python flask rest api cloud deployment",
        "javascript node typescript serverless azure",
        "mlops azure ml model deployment python",
        "cloud security azure key vault python",
        "ai resume screener python machine learning",
        "microsoft azure ai cognitive services python",
        "sql azure data factory pipelines engineer",
        "full stack developer javascript azure python",
        "ai powered chatbot using langchain python",
        "azure kubernetes docker cloud engineer",
        "pytorch deep learning image classification python",

        # Not Relevant (0)
        "c programming basic loops arrays",
        "java beginner oops concepts only",
        "networking fundamentals hardware support",
        "graphic design photoshop illustrator",
        "excel powerpoint word office tools",
        "basic electronics arduino raspberry pi",
        "desktop support troubleshooting windows",
        "embedded systems microcontroller basics",
        "video editing premiere after effects",
        "customer service call center skills",
        "marketing digital social media ads",
        "finance accounting tally ms excel",
        "content writing blogging seo",
        "civil engineering autocad revit",
        "mechanical engineering cad cam",
        "hotel management food service",
        "basic biology chemistry laboratory",
        "electrical wiring motor repair",
        "human resources recruitment onboarding",
        "basic math tutoring primary school",
        "primary teacher lesson planning",
        "fashion design textile basics",
        "animation 2d 3d blender maya",
        "music production audio mixing",
        "photography camera handling basics"
    ],
    "label": [
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
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
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "../models/resume_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")