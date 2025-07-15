import pickle
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast

# --- Page Setup ---
st.set_page_config(page_title="🎬 Semantic Movie Recommender", layout="wide")

# --- Load Pickles ---
movies = pickle.load(open('model/movies.pkl', 'rb'))  # contains movie_id, title, tags
movie_embeddings = pickle.load(open('model/movie_embeddings.pkl', 'rb'))
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_movie_metadata():
    return pickle.load(open('movies_metadata.pkl', 'rb'))

metadata = load_movie_metadata()

# --- Poster Fetching ---
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

# --- Semantic Search Logic ---
def recommend_from_query(query, top_n=5):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], movie_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return top_indices

# --- UI Header ---
st.markdown("<h1 style='text-align: center;'>🎬 Semantic Movie Recommender</h1><hr>", unsafe_allow_html=True)

# --- Text Input ---
query = st.text_input("🔍 Enter your search (theme, actor, company, genre, etc.):", "")

if st.button("Get Recommendations") and query.strip():
    indices = recommend_from_query(query)

    st.markdown("## 🎯 Top Recommendations")
    cols = st.columns(5)
    for i, col in zip(indices, cols):
        movie = movies.iloc[i]
        poster_url = fetch_poster(movie.movie_id)
        with col:
            if st.button(movie.title, key=f"btn_{i}"):
                st.session_state['selected_index'] = i
            st.image(poster_url, use_container_width=True)

# --- Movie Details View ---
if 'selected_index' in st.session_state:
    idx = st.session_state['selected_index']
    base_movie = movies.iloc[idx]
    selected_movie_id = base_movie.movie_id

    # Match metadata
    movie = metadata[metadata['movie_id'] == selected_movie_id].iloc[0]
    poster = fetch_poster(selected_movie_id)

    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>🎞️ {movie.title} (Details)</h2>", unsafe_allow_html=True)
    detail_col1, detail_col2 = st.columns([1, 2])

    with detail_col1:
        st.image(poster, use_container_width=True)

    with detail_col2:
        # Parse genres
        genres = []
        try:
            genres_data = ast.literal_eval(movie.get('genres', '[]'))
            genres = [g['name'] for g in genres_data if 'name' in g]
        except:
            pass

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
