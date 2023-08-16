import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords  # Add this line
from nltk.tokenize import word_tokenize
import streamlit as st

# Load the dataset into a DataFrame
data = pd.read_csv('IMDB_10000.csv')  # Replace with your dataset path

# Remove missing values
data.dropna(inplace=True)

# Tokenize and remove stopwords from the 'plot' column
stop_words = set(stopwords.words('english'))
data['plot'] = data['plot'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['plot'])

# Dimensionality reduction using TruncatedSVD
num_components = 100  # Adjust as needed
svd = TruncatedSVD(n_components=num_components)
reduced_matrix = svd.fit_transform(tfidf_matrix)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(reduced_matrix, reduced_matrix)

# Streamlit app
st.title("Movie Recommendation App")

# User input for movie title
movie_title = st.text_input("Enter a movie title:", "Vikram")

# Rating filter slider
rating_filter = st.slider("Select minimum rating:", 1, 10, 5)  # Adjust the range as needed

# Get recommended movies based on cosine similarity and rating filter
def recommend_movies_cosine_with_filter(movie_title, min_rating):
    if movie_title not in data['title'].values:
        return []
    
    idx = data[data['title'] == movie_title].index[0]
    
    # Get cosine similarity scores for the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get movie indices with high similarity and rating above the filter
    similar_movies = [i for i, score in sim_scores if score > 0.2 and data.iloc[i]['rating'] >= min_rating]
    
    # Get recommended movie titles
    recommended_movies = data.iloc[similar_movies]['title'].tolist()
    
    return recommended_movies

# Get recommended movies
recommended_movies = recommend_movies_cosine_with_filter(movie_title, rating_filter)

# Display recommendations
if recommended_movies:
    st.subheader(f"Recommended movies similar to '{movie_title}' with a minimum rating of {rating_filter}:")
    for movie in recommended_movies:
        st.write(movie)
else:
    st.write("No recommendations found.")
