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

# Load pre-trained SVD model directly
try:
    with open('best_svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Ensure 'best_svd_model.pkl' exists.")
    svd_model = None
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    svd_model = None

# Streamlit UI
st.set_page_config(page_title="Anime Recommendation System", page_icon="🎮")

def emojize(text):
    return emoji.emojize(text)

st.title(emojize(":clapper: Anime Recommendation System"))
st.image("https://storage.googleapis.com/kaggle-datasets-images/571/1094/c633ae058ddaa59f43649caac1748cf4/dataset-card.png", caption="Anime Recommendation System")

tab1, tab2 = st.tabs(["Recommendation", "Team"])

with tab1:
    pickled_data = load_pickled_data()
    df = pickled_data['df1']
    pca_df = pickled_data['df2']
    train = pd.read_csv('train.csv')
    
    model = st.radio(
        emojize(":mag_right: Select Recommendation Method"),
        ['Content-Based (PCA)', 'Collaborative-Based (Rating Predictor - SVD)', 'Collaborative-Based (User Recommendations)', 'Hybrid (PCA + SVD)'],
        index=0
    )
    
    anime_titles = df['name'].unique()

    # PCA-based recommendation function
    def recommend_anime_pca30(input_anime, df, pca_df, top_n=10):
        if input_anime not in df['name'].to_numpy():
            return None
        title_index = df[df['name'] == input_anime].index[0]
        input_pca_vector = pca_df.iloc[title_index, :-1].to_numpy().reshape(1, -1)
        similarities = CS(input_pca_vector, pca_df.iloc[:, :-1]).flatten()
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        recommendations = list(df.iloc[similar_indices]['name'])
        return recommendations, similarities[similar_indices]
    
    if model == "Collaborative-Based (Rating Predictor - SVD)":
        anime_title = st.selectbox(emojize(":film_frames: Select an anime title:"), anime_titles)
        user_id = st.number_input(emojize(":id: Enter User ID:"), min_value=1, step=1)
        
        if st.button(emojize(":robot_face: Predict Rating")):
            with st.spinner('Generating predicted rating...'):
                matching_anime = df.loc[df['name'] == anime_title, 'anime_id']
                if matching_anime.empty:
                    st.write(":no_entry_sign: Anime not found in the dataset.")
                else:
                    anime_id = matching_anime.iloc[0]
                    predicted_rating = round(svd_model.predict(user_id, anime_id).est, 2) if svd_model else None
                    if predicted_rating is not None:
                        st.write(f"📊 **Predicted Rating for {anime_title}:** ⭐ {predicted_rating:.2f}")
                    else:
                        st.write(":warning: Unable to generate a rating. Check model availability.")

    # Hybrid recommendation function
    elif model == "Hybrid (PCA + SVD)":
        anime_title = st.selectbox(emojize(":film_frames: Select an anime title:"), anime_titles)
        user_id = st.number_input(emojize(":id: Enter User ID:"), min_value=1, step=1)
        alpha = st.slider("Weight for Collaborative Filtering (SVD)", 0.0, 1.0, 0.5)
        
        if st.button(emojize(":robot_face: Get Hybrid Recommendations")):
            with st.spinner('Generating recommendations...'):
                pca_recs, pca_scores = recommend_anime_pca30(anime_title, df, pca_df, top_n=50)
                if pca_recs is None:
                    st.write(":no_entry_sign: Anime not found.")
                else:
                    svd_predictions = {}
                    for anime in pca_recs:
                        anime_id = df.loc[df['name'] == anime, 'anime_id'].iloc[0]
                        svd_pred = svd_model.predict(user_id, anime_id).est if svd_model else None
                        if svd_pred is not None:
                            svd_predictions[anime] = round(svd_pred, 2)
                    hybrid_scores = {anime: round(alpha * svd_predictions[anime] + (1 - alpha) * pca_score, 2)
                                     for anime, pca_score in zip(pca_recs, pca_scores) if anime in svd_predictions}
                    sorted_anime = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
                    st.write(":sparkles: Hybrid Recommendations:")
                    for i, (anime, score) in enumerate(sorted_anime[:10], 1):
                        st.write(f"{i}. **{anime}** :star2: (Hybrid Score: {score:.2f})")

with tab2:
    st.write("### Team Members:")
    st.write("**Lebogang** - Team Lead")
    st.write("**Phillip** - Project and Github Manager")
    st.write("**Sanele** - Kaggle Manager")
    st.write("**Tracy** - Trello Manager")
    st.write("**Matlou and Mzwandile** - Canvas Managers")

