# app.py - Streamlit Clustering Viewer
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from wordcloud import WordCloud
import numpy as np
import warnings

st.set_page_config(page_title="Klasterisasi Komentar Netizen", layout="wide")

# --- Load model dan data ---
@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer_tfidf_kmeans.pkl")
    model = joblib.load("model_kmeans.pkl")
    pca = joblib.load("pca_model.pkl") 
    return vectorizer, model, pca

@st.cache_data
def load_data():
    return joblib.load("data_clustered.pkl")

vectorizer, model, pca = load_model()
df = load_data()

# --- Clean function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Load kata kasar dari GitHub ---
@st.cache_data
def load_kata_kasar_from_url():
    url = "https://raw.githubusercontent.com/regalileo/UAS-BIG-DATA/main/kata_kasar.txt"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        lines = response.text.splitlines()
        return set(k.strip() for k in lines if k.strip())
    except Exception as e:
        st.warning(f"Gagal memuat kata kasar dari URL: {e}")
        return set()

kata_kasar = load_kata_kasar_from_url()

def deteksi_ujaran_kebencian(teks):
    return any(kata in teks.split() for kata in kata_kasar)

st.markdown("""
    <h2 style='text-align: center;'>Visualisasi dan Deteksi Klaster Komentar Media Sosial</h2>
    <p style='text-align: center;'>TF-IDF + KMeans</p>
""", unsafe_allow_html=True)

st.write("---")

with st.spinner("Memproses data..."):
    X = vectorizer.transform(df['clean'])
    df['cluster'] = df['cluster'].astype(int)
    centroids = model.cluster_centers_
    centroids_pca = pca.transform(centroids)

# --- Layout utama ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualisasi PCA & Donut Chart")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', s=60, ax=ax1)
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', s=200, marker='X', label='Centroid')
    ax1.set_title("Pemetaan PCA + Centroid Klaster")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    cluster_counts = df['cluster'].value_counts().sort_index()
    labels = [f"Cluster {i}: {n}" for i, n in cluster_counts.items()]
    fig2, ax2 = plt.subplots()
    wedges, texts, autotexts = ax2.pie(cluster_counts, labels=labels,
        autopct="%1.1f%%", startangle=90, wedgeprops={'width': 0.5}, textprops={'fontsize': 9})
    ax2.axis("equal")
    st.pyplot(fig2)

    with st.expander("Statistik Deskriptif per Klaster"):
        df_stats = df.groupby('cluster')[['pca1','pca2']].agg(['mean','std','min','max']).round(2)
        st.dataframe(df_stats)

    with st.expander("Top Fitur Pembeda (Mutual Info)"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_arr = X.toarray()
            mi_scores = mutual_info_classif(X_arr, df['cluster'], discrete_features=True)
            top_idx = np.argsort(mi_scores)[::-1][:10]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_idx]
            top_vals = mi_scores[top_idx]
            top_df = pd.DataFrame({"Fitur": top_words, "MI Score": top_vals})
            st.dataframe(top_df)

with col2:
    st.subheader("Uji Komentar Baru")
    komentar_baru = st.text_area("Masukkan komentar baru:", height=150)

    if st.button("Prediksi Klaster"):
        if komentar_baru.strip() == "":
            st.warning("Komentar tidak boleh kosong!")
        else:
            try:
                clean_komentar = clean_text(komentar_baru)
                st.write("✅ Teks dibersihkan")
                vectorized = vectorizer.transform([clean_komentar])
                st.write("✅ Teks berhasil di-transform")
                cluster_pred = model.predict(vectorized)[0]
                st.success(f"Komentar tersebut masuk dalam **Cluster {cluster_pred}**")

                is_hate = deteksi_ujaran_kebencian(clean_komentar)
                if is_hate:
                    st.error("Komentar ini terdeteksi mengandung **ujaran kebencian** ⚠️")
                else:
                    st.info("Komentar ini **tidak mengandung ujaran kebencian** ✅")
            except Exception as e:
                st.error("❌ Terjadi error saat memproses komentar.")
                st.exception(e)
