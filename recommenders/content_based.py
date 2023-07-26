"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(data, subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Subset of the data
    data_subset = data[:subset_size]

    # Split genre data into individual words.
    data_subset['keyWords'] = data_subset['genres'].str.replace('|', ' ')
    
    return data_subset

def fetch_poster(movie_id):
    response = requests.get()

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    df_sub = data_preprocessing(movies, 15000)

    # Create TF-IDF vectorizer to convert text data to numerical vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_sub['keyWords'])

    # Calculate cosine similarity between movies
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(df_sub['title'])

    # Getting the index of the movie that matches the title
    movies_indices = []
    for movie in movie_list:
        idx = indices[indices == movie].index[0]
        movies_indices.append(idx)

    similar_movies = []
    for index in movies_indices:
        # Get the pairwise similarity scores for the movie
        sim_scores = list(enumerate(cosine_sim[index]))
        # Sort the movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the top 5 similar movies (excluding the input movie itself)
        similar_movies.extend(sim_scores[1:top_n])
    
    # Sort the movies based on similarity scores and remove duplicates
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[0:10]
    
    # Store indices of recommended movies
    recommended_indices = []
    
    for movie in similar_movies:
        index = movie[0]
        if index not in movie_list and index not in recommended_indices:
            recommended_indices.append(index)
    
    # Get the movie titles and genres of the recommended movies
    recommended_movies = df_sub.iloc[recommended_indices]['title']
    
    return recommended_movies