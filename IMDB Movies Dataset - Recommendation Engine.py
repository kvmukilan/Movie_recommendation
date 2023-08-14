#!/usr/bin/env python
# coding: utf-8

# ## Dataset Information
# 
#    Data on Movies from IMDB (Includes Some Television as Well). Movie IDs to help gather much of this data come from one or two Kaggle projects. There is a workflow from original cobbled together spreadsheets to the final product with 27 variables and over 5000 observations. \
#    Content based Filtering

# ## Import Modules

# In[2]:


import pandas as pd
import numpy as np
import re
import nltk
pd.set_option('display.max_columns', None)


# In[2]:






# In[3]:


df = pd.read_csv("IMDB_10000.csv")
df.head()


# In[4]:


len(df)


# In[5]:


df.rename(columns={'desc': 'Plot'}, inplace=True)


# In[6]:


df.head()


# In[7]:


df['genre'] = df['genre'].astype(str)


# In[8]:


df['Plot'][0]


# ## Data Preprocessing

# In[9]:


df['clean_plot'] = df['Plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)) if x else '')
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))


# In[10]:


# tokenize the sentence
df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))
df['clean_plot']


# In[11]:


# remove stopwords
stop_words = nltk.corpus.stopwords.words('english')
plot = []
for sentence in df['clean_plot']:
    temp = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            temp.append(word)
    plot.append(temp)
plot


# In[12]:


df['clean_plot'] = plot


# In[13]:


df['clean_plot']


# In[14]:


df.head()


# In[15]:


df['genre'] = df['genre'].apply(lambda x: x.split(' , '))


# In[56]:





# In[16]:


def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ', ''))
    return temp


# In[17]:


df['genre'] = [clean(x) for x in df['genre']]


# In[18]:





# In[20]:


# combining all the columns data
columns = ['clean_plot', 'genre']
l = []
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)
l


# In[21]:


df['clean_input'] = l
df = df[['title', 'clean_input']]
df.head()


# ## Feature Extraction

# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])


# In[23]:


# create cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(features, features)



# ## Movie Recommendation

# In[25]:


index = pd.Series(df['title'])
index.head()


# In[30]:


def recommend_movies(title):
    movies = []
    idx = index[index == title].index[0]
    # print(idx)
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    # print(top10)
    
    for i in top10:
        movies.append(df['title'][i])
    return movies


# In[32]:


the_movie_u_should_watch = recommend_movies('Vikram')
print(the_movie_u_should_watch)


# In[80]:


index[index == 'Vikram'].index[0]


# In[82]:




# In[56]:





# In[ ]:




