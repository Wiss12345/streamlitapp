#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Lire le fichier Excel
file_path = r'C:\Users\wissa\Downloads\projet2\merged_df.xls'
df = pd.read_csv(file_path)  # Assurez-vous d'utiliser pd.read_excel si le fichier est en .xls

# Afficher les premières lignes du DataFrame pour vérifier le contenu
st.write("Aperçu des données :")
st.write(df.head())

# Convertir release_date en format datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Encodage one-hot des genres
df = pd.get_dummies(df, columns=["genres"], prefix=["genres"])

# Colonnes numériques à normaliser
numeric_columns = ['averageRating', 'numVotes', 'popularity', 'runtime', 'vote_average']

# Normalisation des colonnes numériques
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Préparation des données pour le modèle
features = df.columns.difference(['tconst', 'title', 'cast', 'posters', 'release_date', 'overview', 'production_countries', 'tagline', 'original_language', 'spoken_languages'])

# Sélectionner les caractéristiques pour le modèle
X = df[features]

# Entraîner le modèle KNN
knn_model = NearestNeighbors(n_neighbors=10)
knn_model.fit(X)

def get_genres(row):
    genres = [col.replace("genres_", "") for col in df.columns if col.startswith("genres_") and row[col] == 1]
    return ', '.join(genres)

def recommend_movies(movie_title):
    try:
        # Trouver l'index du film donné
        movie_index = df[df["title"] == movie_title].index[0]
        
        # Trouver les plus proches voisins
        _, indices = knn_model.kneighbors([X.iloc[movie_index]])
        
        # Indices des films recommandés (en excluant le premier, qui est le film lui-même)
        recommended_movies_index = indices[0][1:]
        
        recommendations = []
        for index in recommended_movies_index:
            title = df["title"].iloc[index]
            release_date = df["release_date"].iloc[index]
            genres = get_genres(df.iloc[index])
            poster = df["posters"].iloc[index]
            tagline = df["tagline"].iloc[index]
            overview = df["overview"].iloc[index]
            
            # Vérifier si l'image est un URL ou un chemin valide
            image_path = None
            if isinstance(poster, str):
                if poster.startswith('http'):
                    image_path = poster
                elif os.path.isfile(poster):
                    image_path = poster
            
            recommendations.append({
                "title": title,
                "release_date": release_date,
                "genres": genres,
                "poster": image_path,
                "tagline": tagline,
                "overview": overview
            })
        return recommendations
    except IndexError:
        st.error(f"Le film '{movie_title}' n'a pas été trouvé dans la base de données.")
        return []
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        return []

# Application Streamlit
st.title("Système de Recommandation de Films")

# Input de l'utilisateur
user_input = st.text_input("Entrez le titre du film pour obtenir des recommandations :")

if user_input:
    recommendations = recommend_movies(user_input)
    
    if recommendations:
        st.write(f"Recommandations pour le film '{user_input}':")
        for rec in recommendations:
            st.subheader(rec["title"])
            if rec["poster"]:
                try:
                    st.image(rec["poster"], width=150)
                except Exception as e:
                    st.warning(f"Impossible de charger l'image : {e}")
            st.write(f"Release Date: {rec['release_date'].strftime('%Y-%m-%d') if pd.notnull(rec['release_date']) else 'N/A'}")
            st.write(f"Genres: {rec['genres']}")
            st.write(f"Tagline: {rec['tagline']}")
            st.write(f"Overview: {rec['overview']}")
    else:
        st.write("Aucune recommandation trouvée. Vérifiez le titre du film.")


# In[ ]:




