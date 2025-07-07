import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
movies = pd.read_csv('movies1.csv')
movies['combined_features'] = movies['genres'].str.replace('|', ' ', regex=False)
movies = movies.dropna(subset=['combined_features'])
movies = movies.head(1000)

# Vectorize
tfidf = TfidfVectorizer(stop_words='english', max_features=300)
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Movie index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommender System")

movie_list = movies['title'].sort_values().tolist()
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie)
    st.subheader("Top 5 Recommendations:")
    for movie in recommendations:
        st.write("🎥", movie)
