import pickle
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast

# --- Page Setup ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# --- Load Pickles for Recommendations ---
movies = pickle.load(open('model/movies.pkl', 'rb'))  # contains movie_id, title, tags
movie_embeddings = pickle.load(open('model/movie_embeddings.pkl', 'rb'))
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Load Full Movie Metadata (from GitHub) ---
@st.cache_data
def load_movie_metadata():
    return pickle.load(open('movies_metadata.pkl', 'rb'))

metadata = load_movie_metadata()

# --- TMDB Poster Fetch ---
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path', '')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# --- Recommendation Logic ---
def recommend(query, top_n=5):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], movie_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return top_indices

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🎬 Semantic Movie Recommender</h1><hr>", unsafe_allow_html=True)

# --- Input Query Box ---
query = st.text_input("🔍 Enter a description, actor, genre, theme, etc.")

if st.button("🎯 Get Recommendations") and query:
    indices = recommend(query)

    st.markdown("## 🎥 Recommended Movies")

    for idx in indices:
        base_movie = movies.iloc[idx]
        movie_id = base_movie.movie_id
        movie = metadata[metadata['movie_id'] == movie_id].iloc[0]
        poster = fetch_poster(movie_id)

        # Parse genres
        genres = []
        try:
            genres_data = ast.literal_eval(movie.get('genres', '[]'))
            genres = [g['name'] for g in genres_data if 'name' in g]
        except:
            genres = []

        # Display movie card
        with st.container():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(poster, use_container_width=True)

            with col2:
                st.markdown(f"### 🎬 {movie['title']}")
                st.markdown(f"**📝 Tagline**: {movie.get('tagline', '-')}")
                st.markdown(f"**🧾 Overview**: {movie.get('overview', '-')}")
                st.markdown(f"**🎭 Genres**: {', '.join(genres) if genres else '-'}")
                st.markdown(f"**🌐 Language**: {movie.get('original_language', '-')}")
                st.markdown(f"**📅 Release Date**: {movie.get('release_date', '-')}")
                st.markdown(f"**⏱️ Runtime**: {movie.get('runtime', '-')} mins")
                st.markdown(f"**⭐ Popularity**: {movie.get('popularity', '-')}")
                st.markdown(f"**🌟 Vote Avg**: {movie.get('vote_average', '-')}")
                st.markdown(f"**👥 Vote Count**: {movie.get('vote_count', '-')}")
                st.markdown(f"**💰 Budget**: ${movie.get('budget', '-')}")
                st.markdown(f"**💵 Revenue**: ${movie.get('revenue', '-')}")
                st.markdown(f"**📈 Status**: {movie.get('status', '-')}")
                if movie.get('homepage', ''):
                    st.markdown(f"**🔗 Homepage**: [Visit]({movie.get('homepage')})")
