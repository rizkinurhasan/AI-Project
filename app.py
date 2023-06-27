pip install beautifulsoup4
import streamlit as st
import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fetch_poster_from_api(movie_title):
    api_key = "29624157a53a561f1262f28ce77c41c5"
    formatted_title = movie_title.lower().replace(" ", "+")
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={formatted_title}"
    response = requests.get(url)
    data = response.json()
    results = data.get('results')

    if results:
        poster_path = results[0].get('poster_path')
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return full_path

    return None

def fetch_company_images(movie_title):
    api_key = "29624157a53a561f1262f28ce77c41c5"
    formatted_title = movie_title.lower().replace(" ", "+")
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={formatted_title}"
    response = requests.get(url)
    data = response.json()
    results = data.get('results')

    if results:
        movie_id = results[0].get('id')
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer 29624157a53a561f1262f28ce77c41c5"
        }
        response = requests.get(url, headers=headers)
        return response.json()

    return None

# Membaca file CSV
df = pd.read_csv('Ai.csv')

# Mengonversi DataFrame menjadi kamus
movies_dict = df.to_dict(orient='records')

# Menyimpan kamus ke dalam file pickle
with open('Ai.pkl', 'wb') as file:
    pickle.dump(movies_dict, file)

# Load the movie data from pickle
movies_dict = pickle.load(open('Ai.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

st.title('Movie Recommender System')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['title'])

# Compute the cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movie_posters = []

    for i in movies_list:
        # fetch the movie poster
        movie_title = movies.iloc[i[0]]['title']
        recommended_movies.append(movie_title)
        recommended_movie_posters.append(fetch_poster_from_api(movie_title))

    return recommended_movies, recommended_movie_posters

selected_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if selected_movie_name:
    if st.button('Show Recommendation'):
        names, posters = recommend(selected_movie_name)

        # Menghitung jumlah kolom berdasarkan jumlah rekomendasi
        num_recommendations = len(names)
        num_columns = min(num_recommendations, 5)
        columns = st.columns(num_columns)

        # Menampilkan judul dan gambar secara otomatis pada setiap kolom
        for i in range(num_recommendations):
            with columns[i % num_columns]:
                if posters[i]:
                    st.image(posters[i], caption='', width=100)
                    st.subheader(names[i])
                else:
                    st.write("No poster available")
                    st.subheader(names[i])
