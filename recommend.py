import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies dataset
movies = pd.read_csv('movies.csv')

# Vectorize genres
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])

# Compute cosine similarity
similarity = cosine_similarity(genre_matrix)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        print(f"❌ Movie '{movie_title}' not found in dataset!")
        return
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies
    print(f"\n🎬 Top 5 recommendations for '{movie_title}':")
    for i, score in sim_scores:
        print(f"- {movies['title'][i]}")

# Main program
if __name__ == "__main__":
    print("🎥 Welcome to the Movie Recommendation System!")
    movie = input("Enter a movie title: ")
    recommend(movie)
