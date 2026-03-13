"""
Create demo models for testing the cyberbullying detection platform.
This script creates simple working models when the original .pkl files are corrupted.
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

class SimpleTextCleaner:
    def transform(self, texts):
        cleaned = []
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            text = ' '.join(text.split())
            cleaned.append(text)
        return cleaned

# Sample training data
X_train = [
    "you are stupid and ugly",
    "i hate you so much",
    "you should die loser",
    "kill yourself idiot",
    "you are worthless trash",
    "hello how are you",
    "have a great day",
    "nice work on the project",
    "thank you for your help",
    "lets meet tomorrow"
]

y_train = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 = bullying, 0 = safe

# Create and train models
print("Creating text cleaner...")
cleaner = SimpleTextCleaner()

print("Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_cleaned = cleaner.transform(X_train)
X_vectorized = vectorizer.fit_transform(X_cleaned)

print("Training model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_vectorized, y_train)

# Save models
print("\nSaving models...")
with open("../cyberbullying_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✓ Saved cyberbullying_model.pkl")

with open("../tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✓ Saved tfidf_vectorizer.pkl")

with open("../text_cleaner.pkl", "wb") as f:
    pickle.dump(cleaner, f)
print("✓ Saved text_cleaner.pkl")

print("\n✓ All models created successfully!")
print("\nTest prediction:")
test_text = ["you are stupid"]
test_cleaned = cleaner.transform(test_text)
test_vec = vectorizer.transform(test_cleaned)
prediction = model.predict(test_vec)[0]
proba = model.predict_proba(test_vec)[0]
print(f"Text: {test_text[0]}")
print(f"Prediction: {'Cyberbullying' if prediction == 1 else 'Safe'}")
print(f"Confidence: {max(proba):.2f}")
