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

# --- Load Full Movie Metadata (Merged Dataset Pickle) ---
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

# --- Semantic Recommendation Function ---
def recommend(query, top_n=5):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], movie_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return top_indices

# --- App Header ---
st.markdown("<h1 style='text-align: center;'>🎬 Semantic Movie Recommender</h1><hr>", unsafe_allow_html=True)

# --- Query Input ---
user_query = st.text_input("🔍 Enter a movie genre, theme, actor, or description:")

if st.button("🎯 Get Recommendations") and user_query:
    indices = recommend(user_query)

    for idx in indices:
        movie = movies.iloc[idx]
        selected_movie_id = movie.movie_id

        # Metadata lookup
        details = metadata[metadata['movie_id'] == selected_movie_id].iloc[0]
        poster_url = fetch_poster(selected_movie_id)

        # Genres parsing
        genres = []
        try:
            genres_data = ast.literal_eval(details.get('genres', '[]'))
            genres = [g['name'] for g in genres_data if 'name' in g]
        except:
            genres = []

        # Card Layout
        with st.container():
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'>{details.title}</h2>", unsafe_allow_html=True)

            top_col1, top_col2 = st.columns([1, 2])
            with top_col1:
                st.image(poster_url, use_container_width=True)

            with top_col2:
                st.markdown(f"**📝 Tagline**: {details.get('tagline', '-')}")
                st.markdown(f"**🧾 Overview**: {details.get('overview', '-')}")

                # Two-column metadata view
                meta_col1, meta_col2 = st.columns(2)

                with meta_col1:
                    st.markdown(f"**🎭 Genres**: {', '.join(genres) if genres else '-'}")
                    st.markdown(f"**🌐 Language**: {details.get('original_language', '-')}")
                    st.markdown(f"**📅 Release Date**: {details.get('release_date', '-')}")
                    st.markdown(f"**⏱️ Runtime**: {details.get('runtime', '-')} mins")
                    st.markdown(f"**⭐ Popularity**: {details.get('popularity', '-')}")  
                    st.markdown(f"**🌟 Vote Avg**: {details.get('vote_average', '-')}")

                with meta_col2:
                    st.markdown(f"**👥 Vote Count**: {details.get('vote_count', '-')}")
                    st.markdown(f"**💰 Budget**: ${details.get('budget', '-')}")
                    st.markdown(f"**💵 Revenue**: ${details.get('revenue', '-')}")
                    st.markdown(f"**📈 Status**: {details.get('status', '-')}")
                    homepage = details.get('homepage', '')
                    if homepage:
                        st.markdown(f"**🔗 Homepage**: [Visit]({homepage})")
