import streamlit as st
import dill
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as CS
import emoji

# Load pickled data
@st.cache_resource
def load_pickled_data():
    try:
        with open("function_dict4.pkl", "rb") as f:
            pickled_data = dill.load(f)
        return pickled_data
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

# Load pre-trained SVD model
@st.cache_resource
def load_svd_model():
    try:
        with open('best_svd_model.pkl', 'rb') as f:
            svd_model = pickle.load(f)
        return svd_model
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'best_svd_model.pkl' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Anime Recommendation System", page_icon="🎬")

# Use emoji package for shortcodes
def emojize(text):
    return emoji.emojize(text)

st.title(emojize(":clapper: Anime Recommendation System"))

# Add image below the title
st.image("https://storage.googleapis.com/kaggle-datasets-images/571/1094/c633ae058ddaa59f43649caac1748cf4/dataset-card.png", caption="Anime Recommendation System")

# Create a team tab
tab1, tab2 = st.tabs(["Recommendation", "Team"])

# Recommendation tab
with tab1:
    # Load data
    pickled_data = load_pickled_data()
    df = pickled_data['df1']
    pca_df = pickled_data['df2']
    train = pd.read_csv('train.csv')

    # Recommendation method selection
    model = st.radio(
        emojize(":mag_right: Select Recommendation Method"),
        ['Content-Based (PCA)', 'Collaborative-Based (Rating Predictor - SVD)', 'Collaborative-Based (User Recommendations)', 'Hybrid (PCA + SVD)'],
        index=0
    )

    # User Input
    anime_titles = df['name'].unique()

    # PCA-based recommendation function
    def recommend_anime_pca30(input_anime, df, pca_df, top_n=10):
        if input_anime not in df['name'].to_numpy():
            return None

        title_index = df[df['name'] == input_anime].index[0]
        input_pca_vector = pca_df.iloc[title_index, :-1].to_numpy().reshape(1, -1)

        similarities = CS(input_pca_vector, pca_df.iloc[:, :-1])
        similarities = similarities.flatten()

        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        recommendations = list(df.iloc[similar_indices]['name'])

        return recommendations, similarities[similar_indices]

    # Collaborative-based rating predictor function
    def get_predicted_rating(input_anime, user_id, svd_model):
        if svd_model is None:
            st.error("The recommendation model is not available. Please check the loading process.")
            return None
        try:
            return round(svd_model.predict(user_id, input_anime).est, 2)
        except Exception as e:
            st.error(f"Error predicting rating: {str(e)}")
            return None

    # Collaborative-based recommendation function
    def recommend_anime_for_user(user_id, svd_model, df, pca_df, top_n=10, alpha=0.0):
        anime_ids = df['anime_id'].unique()
        anime_names = df.set_index('anime_id')['name'].to_dict()

        svd_predictions = {anime_id: svd_model.predict(user_id, anime_id).est for anime_id in anime_ids}
        svd_df = pd.DataFrame(list(svd_predictions.items()), columns=['anime_id', 'svd_rating'])
        svd_df['name'] = svd_df['anime_id'].map(anime_names)
        svd_df = svd_df.sort_values(by='svd_rating', ascending=False).head(50)

        hybrid_scores = {}
        for _, row in svd_df.iterrows():
            anime = row['name']
            pca_recs, pca_scores = recommend_anime_pca30(anime, df, pca_df, top_n=1)
            if pca_recs:
                pca_score = pca_scores[0]
                hybrid_scores[anime] = round(alpha * row['svd_rating'] + (1 - alpha) * pca_score, 2)

        sorted_anime = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [anime for anime, _ in sorted_anime[:top_n]]

    # Streamlit UI for Content-Based (PCA)
    if model == "Content-Based (PCA)":
        anime_title = st.selectbox(emojize(":film_frames: Select an anime title:"), anime_titles)
        if st.button(emojize(":robot_face: Get Content-Based Recommendations")):
            with st.spinner('Generating recommendations...'):
                recommendations, _ = recommend_anime_pca30(anime_title, df, pca_df, top_n=10)
            if recommendations is None:
                st.write(":no_entry_sign: Anime not found.")
            else:
                st.write(":sparkles: Content-Based Recommendations:")
                for i, anime in enumerate(recommendations, 1):
                    st.write(f"{i}. {anime} :tv:")

    # Streamlit UI for Collaborative Filtering (Rating Predictor SVD)
    elif model == "Collaborative-Based (Rating Predictor - SVD)":
        anime_title = st.selectbox(emojize(":film_frames: Select an anime title:"), anime_titles)
        user_id = st.number_input(emojize(":id: Enter User ID:"), min_value=1, step=1)
        if st.button(emojize(":robot_face: Predict Rating")):
            with st.spinner('Generating predicted rating...'):
                svd_model = load_svd_model()
                if svd_model is None:
                    st.write(":no_entry_sign: Unable to load the recommendation model.")
                else:
                    matching_anime = df.loc[df['name'] == anime_title, 'anime_id']
                    if matching_anime.empty:
                        st.write(":no_entry_sign: Anime not found in the dataset.")
                    else:
                        anime_id = matching_anime.iloc[0]
                        predicted_rating = get_predicted_rating(user_id, anime_id, svd_model)
                        if predicted_rating is not None:
                            st.write(f"📊 **Predicted Rating for {anime_title}:** ⭐ {predicted_rating:.2f}")

# Team tab
with tab2:
    st.write("### Team Members:")
    st.write("**Lebogang** - Team Lead")
    st.write("**Phillip** - Project and GitHub Manager")
    st.write("**Sanele** - Kaggle Manager")
    st.write("**Tracy** - Trello Manager")
    st.write("**Matlou and Mzwandile** - Canvas Managers")
