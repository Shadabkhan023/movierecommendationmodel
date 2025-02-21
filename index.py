import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load the trained model
try:
    model_path = "model.pkl"
    df, tfidf_vectorizer, nn_model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    df, tfidf_vectorizer, nn_model = None, None, None

def fetch_poster(movie_id):
    api_key = os.getenv("TMDB_API_KEY", "c7ec19ffdd3279641fb606d19ceb9bb1")  # Use environment variable
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get("poster_path", "")  
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return "https://via.placeholder.com/150" 

def recommend_movies(movie_name, n=5):
    if not tfidf_vectorizer or not nn_model:
        return []
    
    query_vec = tfidf_vectorizer.transform([movie_name])
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=n)
    
    recommendations = []
    for index in indices[0]:
        movie_title = df.iloc[index]["title"]
        movie_id = df.iloc[index]["id"]  
        poster_url = fetch_poster(movie_id)
        recommendations.append({"title": movie_title, "poster": poster_url})
    
    return recommendations

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    movie_name = data.get("movie_name", "")
    recommendations = recommend_movies(movie_name)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use environment variable for Render
    app.run(host="0.0.0.0", port=port, debug=True)
