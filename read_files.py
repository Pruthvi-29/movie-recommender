import pandas as pd

# Load the data
movies = pd.read_csv('movies1.csv')

# Create combined features from genres
movies['combined_features'] = movies['genres'].str.replace('|', ' ', regex=False)

# Drop rows with missing genre info to minimize storage usage
movies = movies.dropna(subset=['combined_features'])

# TEMP FIX: Use only first 1000 movies to reduce memory usage
movies = movies.head(1000)

# Vectorization with limited features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=300)
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Compute cosine similarity
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to their index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the given movie
    idx = indices.get(title)
    
    # If movie not found
    if idx is None:
        return "Movie not found in the dataset."

    # Get similarity scores for all movies compared to the input movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top 5 most similar movies (excluding the first one)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 5 movie titles
    return movies['title'].iloc[movie_indices]


print(get_recommendations("Toy Story (1995)"))

