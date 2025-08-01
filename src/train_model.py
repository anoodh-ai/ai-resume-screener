import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample training data (later can expand)
data = {
    "text": [
        "python machine learning sql azure",
        "c programming basic python",
        "javascript azure html css",
        "python sql machine learning javascript azure",
        "basic java html"
    ],
    "label": [1, 0, 1, 1, 0]  # 1 = relevant, 0 = not relevant
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