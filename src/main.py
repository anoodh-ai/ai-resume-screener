import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Load sample data (temporary CSV or hardcoded)
data = [
    ("Python developer with ML experience", 1),
    ("Sales manager experience, no tech background", 0)
]

# Step 2: Convert to DataFrame
df = pd.DataFrame(data, columns=["resume_text", "selected"])

# Step 3: Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume_text"])
y = df["selected"]

# Step 4: Train simple model
model = LogisticRegression()
model.fit(X, y)

# Step 5: Save model
with open("../models/resume_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved successfully!")