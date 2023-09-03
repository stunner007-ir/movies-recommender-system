import pandas as pd
import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


movies['tags'] = movies['tags'].apply(stem)


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?language=en-US".format(movie_id)

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjYWFiOTg2NTc0NjhmNTRkYzQyMWViYTA4NDExZmFmMCIsInN1YiI6IjY0ZjNkZjg3OTdhNGU2MDBmZWE5ZjQ4OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.4mFK0vl__kYUyWrPwwqeC5XtBIz_63pfDkYY3h5kZbs"
    }

    response = requests.get(url, headers=headers)
    data = response.json()
    print(response)
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


cv2 = CountVectorizer(max_features=5000, stop_words='english')
vectors2 = cv2.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors2)
sorted(similarity[0], reverse=True)


def recommends(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = {}
    for i in movies_list:
        recommended_movies[movies.iloc[i[0]].title] = movies.iloc[i[0]].movie_id

    return recommended_movies


def dictionary_to_lists(input_dict):
    keys_list = []
    values_list = []

    for key, value in input_dict.items():
        keys_list.append(key)
        values_list.append(fetch_poster(value))

    return keys_list, values_list


st.title('Movie Recommender System')

selected_movie_name = st.selectbox('Select Movie', movies['title'].values)

ans = recommends(selected_movie_name)

if st.button('Recommend'):
    names, posters = dictionary_to_lists(ans)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.header(names[0])
        st.image(posters[0])
    with col2:
        st.header(names[1])
        st.image(posters[1])
    with col3:
        st.header(names[2])
        st.image(posters[2])
    with col4:
        st.header(names[3])
        st.image(posters[3])
    with col5:
        st.header(names[4])
        st.image(posters[4])
