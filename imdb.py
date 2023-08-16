import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your DataFrame from the CSV file
df = pd.read_csv("IMDB_10000.csv")
df.rename(columns={'desc': 'Plot'}, inplace=True)
df['genre'] = df['genre'].astype(str)

# Data Preprocessing
df['clean_plot'] = df['Plot'].str.lower().replace('[^a-zA-Z]', ' ', regex=True).str.replace('\s+', ' ', regex=True)
stop_words = set(nltk.corpus.stopwords.words('english'))
df['clean_plot'] = df['clean_plot'].apply(lambda sentence: [word for word in str(sentence).split() if word not in stop_words and len(word) >= 3])

# Genre preprocessing
df['genre'] = df['genre'].apply(lambda x: [word.lower().replace(' ', '') for word in x.split(',')])  # Corrected here

df['clean_input'] = df.apply(lambda row: ' '.join(row['clean_plot']) + ' ' + ' '.join(row['genre']), axis=1)

# Feature Extraction
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])
cosine_sim = cosine_similarity(features, features)

# Movie Recommendation
index = pd.Series(df.index, index=df['title'])

def recommend_movies(title, min_rating=5, genre=None):
    movies = []
    idx = index[title]
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    
    for i in top10:
        if df['rating'][i] >= min_rating and (genre is None or genre in df['genre'][i]):
            movies.append(df['title'][i])
    return movies

# Example usage
the_movies_you_should_watch = recommend_movies('Vikram', min_rating=5, genre='action')
print(the_movies_you_should_watch)
