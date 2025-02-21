import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Load the dataset
df = pd.read_csv("movie.csv")

# Check dataset structure
print(df.head())

# Ensure 'title' column exists
if 'title' not in df.columns:
    raise ValueError("Dataset must contain a 'title' column!")

# Train TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])

# Train Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(tfidf_matrix)

# Save the model
joblib.dump((df, tfidf_vectorizer, nn_model), "model.pkl")

print("✅ Model training & saving complete!")
