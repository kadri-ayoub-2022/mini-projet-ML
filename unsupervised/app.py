import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# =========================
# Chargement des modèles
# =========================
dbscan_model = joblib.load("dbscan_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler_model.pkl")
pca = joblib.load("pca.pkl")

# =========================
# Dataset
# =========================
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)

# =========================
# Interface Streamlit
# =========================
st.title("Clustering interactif avec modèles PKL")

model_choice = st.selectbox(
    "Choisir le modèle :",
    ["DBSCAN", "KMeans avec PCA"]
)

# =========================
# Entrée utilisateur
# =========================
st.subheader("Entrer un nouveau point")
input_data = {}

for col in X.columns:
    val = st.number_input(col, value=float(X[col].mean()))
    input_data[col] = val

new_point = np.array([list(input_data.values())])

# =========================
# Standardisation
# =========================
X_scaled = scaler.transform(X)
new_point_scaled = scaler.transform(new_point)

# =========================
# ----- DBSCAN -----
# =========================
if model_choice == "DBSCAN":
    labels = dbscan_model.fit_predict(X_scaled)

    # Trouver le cluster du nouveau point (approximation par voisin le plus proche)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(X_scaled)
    dist, idx = neigh.kneighbors(new_point_scaled)

    new_label = labels[idx[0][0]]

    if new_label == -1:
        st.error("❌ Le point est considéré comme BRUIT")
    else:
        st.success(f"✅ Le point appartient au cluster : {new_label}")

    # Visualisation (2 premières features)
    plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
    plt.scatter(
        new_point_scaled[:, 0],
        new_point_scaled[:, 1],
        marker="X",
        s=150
    )
    plt.title("DBSCAN - Nouveau point")
    st.pyplot(plt)

# =========================
# ----- KMEANS + PCA -----
# =========================
elif model_choice == "KMeans avec PCA":
    X_pca = pca.transform(X_scaled)
    new_point_pca = pca.transform(new_point_scaled)
    X_pca = X_pca[:, :2]
    new_point_pca = new_point_pca[:, :2]

    print(X_pca.shape)
    labels = kmeans_model.predict(X_pca)
    new_label = kmeans_model.predict(new_point_pca)[0]

    st.success(f"✅ Le point appartient au cluster : {new_label}")

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.scatter(
        new_point_pca[:, 0],
        new_point_pca[:, 1],
        marker="X",
        s=150
    )
    plt.title("KMeans + PCA - Nouveau point")
    st.pyplot(plt)
