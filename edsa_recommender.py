"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Data visualization dependencies
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model, data_preprocessing

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset for EDA (Assuming 'movies.csv' contains columns like 'title', 'genres', 'ratings', etc.)
movies = pd.read_csv('resources/data/movies.csv')
movies_imdb_df=pd.read_csv('resources/data/movies_imdb.csv')
imdb=pd.read_csv('resources/data/imdb.csv')
ratings = pd.read_csv('resources/data/ratings.csv')

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

st.set_page_config(page_title="CineFlix", page_icon="::", layout="wide")

bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://images.unsplash.com/photo-1595769816263-9b910be24d5f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1179&q=80');
background-size: cover;
background-position: top center;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.unsplash.com/photo-1616530940355-351fabd9524b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=435&q=80');
background-size: cover;
background-position: top;
}
</style>
"""
st.markdown(bg_img, unsafe_allow_html=True)
st.title('CineFlix')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Exploratory Data Analysis", "Solution Overview", "Business Pitch"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        # st.write('### Explore Data Science Academy Unsupervised Learning')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[1:50])
        movie_2 = st.selectbox('Second Option',title_list[100:150])
        movie_3 = st.selectbox('Third Option',title_list[1000:1150])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write(""""
                 This movie recommender system was developed as part of the EXPLORE Data Science Academy Unsupervised Predict.
                 Our winning approach combines both content-based and collaborative filtering algorithms to provide personalized movie recommendations.
                 The content-based filtering algorithm leverages the attributes and features of movies (e.g., genre, director, cast) to suggest similar movies based on the user's favorite movies.
                 On the other hand, the collaborative-based filtering algorithm identifies patterns in user-item interactions to recommend movies based on the preferences of similar users.
                 By combining these two approaches, we create a hybrid recommender system that provides diverse and accurate movie recommendations for each user.\n\nThe collaborative-based filtering algorithm employs matrix factorization techniques such as Singular Value Decomposition (SVD) to handle the sparsity of the user-item interaction matrix and extract latent features that represent user preferences and movie characteristics.
                 The content-based filtering algorithm uses natural language processing techniques to analyze the movie descriptions and extract meaningful features that contribute to the movie's content representation. Our approach allows us to address the cold-start problem by using content-based recommendations for new users with limited interaction data and gradually transitioning to collaborative-based recommendations as more user data becomes available. The final list of top-10 movie recommendations is generated by combining the results of both algorithms and presenting them to the user. With this hybrid recommender system, we aim to deliver a personalized and engaging movie-watching experience for every user, encouraging them to discover new movies that match their preferences.
                 """)
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    elif page_selection == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")
        # Add your EDA code and visualizations here
        # Show the first few rows of the dataset
        st.write("Preview of the Dataset:")
        st.dataframe(movies.head())

        # Summary statistics
        st.write("Summary Statistics:")
        st.dataframe(movies.describe())

        # Distribution of movie genres
        st.title("Distribution of Movie Genres")
        genre_counts = movies['genres'].str.split('|', expand=True).stack().value_counts()
        
        # Create a DataFrame for the plot
        genre_counts_df = pd.DataFrame({'Genre': genre_counts.index, 'Count': genre_counts.values})


        st.write('Bar chart showing the count of each genre in the dataset.')

        # Create the plot using Plotly Express
        fig = px.bar(genre_counts_df, x='Genre', y='Count', title='Movie Genre Counts')
        st.plotly_chart(fig)

        # Occurrence of plot keywords
        st.title("Occurrence of plot_keywords")
        plot_keywords = movies_imdb_df['plot_keywords'].str.split('|', expand=True).stack().value_counts()
        
        # Create a DataFrame for the plot
        plot_keywords_df = pd.DataFrame({'keyword': plot_keywords.index, 'Count': plot_keywords.values})


        st.write('Bar chart showing the count of each genre in the dataset.')

        # Create the plot using Plotly Express
        fig2 = px.bar(plot_keywords_df[:10], x='keyword', y='Count', title='Movie Plot Keyword Counts')
        st.plotly_chart(fig2)

        # Distribution of Ratings (Added plot using sns)
        st.title("Distribution of Ratings")
        fig = go.Figure(data=[go.Histogram(x=ratings['rating'])])
        st.plotly_chart(fig)  
        
        # Distribution of Runtime (Added plot using sns)
        st.title("Distribution of Runtime")
        # Filter the data to keep runtimes less than 400
        filtered_imdb = imdb[imdb['runtime'] < 400]

        # Create the Plotly figure
        fig = go.Figure(data=[go.Histogram(x=filtered_imdb['runtime'])])

        # Customize the layout if needed
        fig.update_layout(
            title='Movie Runtimes',
            xaxis_title='Runtime',
            yaxis_title='Count',
        )

        # Create the Streamlit app
        
        st.write('Histogram showing the distribution of movie runtimes (less than 400 minutes).')

        # Display the plot using Streamlit and Plotly
        st.plotly_chart(fig)

        # Calculate the movie count for each director
        director_counts = imdb['director'].value_counts()

        # Select the top 10 directors based on the movie count
        top_10_directors = director_counts.head(10)

        # Filter the data to keep movies directed by the top 10 directors
        filtered_imdb = imdb[imdb['director'].isin(top_10_directors.index)]

        # Create the Plotly figure
        fig = go.Figure(data=[go.Histogram(x=filtered_imdb['director'])])

        # Customize the layout if needed
        fig.update_layout(
            title='Distribution of Movies Across Top 10 Directors',
            xaxis_title='Director',
            yaxis_title='Movie Count',
        )

        # Create the Streamlit app
        st.title("Distribution of movies across top 10 directors")
        st.write("Histogram showing the distribution of movies across the top 10 directors.")

        # Display the plot using Streamlit and Plotly
        st.plotly_chart(fig)



    elif page_selection == "Business Pitch":
        st.title("Business Pitch")
        st.write("""
                 Our movie recommender system presents a lucrative opportunity for businesses to enhance user engagement and satisfaction on their streaming platforms.
                 Here's why our solution stands out:

                 1. Personalized Recommendations: Our hybrid recommender system combines the strengths of content-based and collaborative-based filtering, providing users with highly personalized movie suggestions based on their unique preferences.

                 2. Improved User Retention: By offering accurate and diverse movie recommendations, users are more likely to spend more time on the platform, leading to increased user retention and reduced churn rate.

                 3. Enhanced User Experience: Users appreciate platforms that understand their tastes and interests. Our system ensures that users receive relevant and exciting movie recommendations, leading to higher user satisfaction.

                 4. Boosted Revenue: Satisfied and engaged users are more likely to make repeat purchases and subscribe to premium services, resulting in increased revenue for the platform.

                 5. Scalable and Adaptable: Our solution can be easily scaled to accommodate a growing user base and adapted to handle various types of media content beyond movies, such as TV shows and documentaries.

                 Partnering with us to integrate this powerful recommender system into your platform will undoubtedly lead to a significant boost in user engagement, retention, and revenue. Let's work together to revolutionize the movie-watching experience for your users!
                 """)


if __name__ == '__main__':
    main()
