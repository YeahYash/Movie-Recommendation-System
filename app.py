from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and prepare data
movies = pd.read_csv('movies.csv')
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])
similarity = cosine_similarity(genre_matrix)

def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    return [movies['title'][i[0]] for i in sim_scores]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie = request.form['movie']
        recommendations = recommend(movie)
        return render_template('result.html', movie=movie, recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
