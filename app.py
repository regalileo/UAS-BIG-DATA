# app.py - Tampilan Klastering Streamlit yang Diringkas dan Rapi

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
import numpy as np
import warnings

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from wordcloud import WordCloud

# Konfigurasi halaman
st.set_page_config(page_title="Klasterisasi Komentar Netizen", layout="wide")

# Memuat model
@st.cache_resource
def load_model():
    try:
        vectorizer = joblib.load("vectorizer_tfidf_kmeans.pkl")
        model = joblib.load("model_kmeans.pkl")
        pca = joblib.load("pca_model.pkl")
        return vectorizer, model, pca
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# Memuat data
@st.cache_data
def load_data():
    try:
        df = joblib.load("data_clustered.pkl")
        df['cluster'] = df['cluster'].astype(int)
        if 'pca1' not in df or 'pca2' not in df:
            if 'clean' in df:
                X = vectorizer.transform(df['clean'])
                pca_result = pca.transform(X)
                df['pca1'], df['pca2'] = pca_result[:, 0], pca_result[:, 1]
            else:
                st.error("Kolom clean tidak ditemukan.")
                st.stop()
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# Pembersihan teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Memuat kata kasar
@st.cache_data
def load_kata_kasar_from_url():
    url = "https://raw.githubusercontent.com/regalileo/UAS-BIG-DATA/main/kata_kasar.txt"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return set(k.strip() for k in r.text.splitlines() if k.strip())
    except:
        return set()

kata_kasar = load_kata_kasar_from_url()

def deteksi_ujaran_kebencian(teks):
    return any(kata in teks.split() for kata in kata_kasar)

# Interpretasi klaster
cluster_interpretations = {
    0: "Diskusi Umum dan Netral",
    1: "Kritik dan Tuntutan Hukum",
    2: "Dukungan dan Doa Kesehatan untuk Jokowi",
    3: "Fokus pada Najwa Shihab ('Mbak Nana')",
    4: "Diskusi tentang Program 'Mata Najwa'",
    5: "Doa dan Harapan Kesehatan yang Kuat untuk Jokowi",
    6: "Komentar tentang Gerakan Kaki Jokowi (Tremor)"
}

# Judul aplikasi
st.markdown("""
    <h3 style='text-align: center;'>Visualisasi & Deteksi Klaster Komentar</h3>
    <p style='text-align: center; font-size: 14px;'>Aplikasi KMeans + TF-IDF + PCA</p>
""", unsafe_allow_html=True)

st.write("---")

vectorizer, model, pca = load_model()
df = load_data()

# Layout Tab

with st.spinner("Memuat visualisasi dan fitur analitik..."):
    X = vectorizer.transform(df['clean'])
    centroids_pca = pca.transform(model.cluster_centers_)
    new_tabs = st.tabs(["PCA", "Distribusi", "Statistik", "Fitur", "Komentar Baru", "WordCloud"])

    with new_tabs[0]:
        st.subheader("Visualisasi PCA Klaster Komentar")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='viridis', ax=ax)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=150, edgecolor='black')
        ax.set_xlabel("Komponen 1"); ax.set_ylabel("Komponen 2")
        ax.set_title("Peta PCA Komentar dan Centroid")
        st.pyplot(fig)

    with new_tabs[1]:
        st.subheader("Distribusi Komentar per Klaster")
        cluster_counts = df['cluster'].value_counts().sort_index()
        labels = [f"Klaster {i}: {cluster_interpretations.get(i)} ({n})" for i, n in cluster_counts.items()]
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax2.pie(cluster_counts, autopct='%1.1f%%', startangle=90, wedgeprops={'width':0.4})
        ax2.axis('equal')
        ax2.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig2)

    with new_tabs[2]:
        st.subheader("Statistik PCA per Klaster")
        stats = df.groupby('cluster')[['pca1', 'pca2']].agg(['mean', 'std', 'min', 'max']).round(3)
        st.dataframe(stats)

    with new_tabs[3]:
        st.subheader("Top Fitur Pembeda (Mutual Information)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_arr = X.toarray()
            mi_scores = mutual_info_classif(X_arr, df['cluster'], discrete_features=True)
            top_idx = np.argsort(mi_scores)[::-1][:10]
            top_words = vectorizer.get_feature_names_out()[top_idx]
            top_vals = mi_scores[top_idx]
            mi_df = pd.DataFrame({'Fitur': top_words, 'Skor MI': top_vals.round(4)})
            st.dataframe(mi_df)

    with new_tabs[4]:
        st.subheader("Prediksi Komentar Baru")
        komentar = st.text_area("Masukkan komentar:")
        if st.button("Klasifikasikan") and komentar.strip():
            clean_komentar = clean_text(komentar)
            X_new = vectorizer.transform([clean_komentar])
            pred = model.predict(X_new)[0]
            st.success(f"Klaster: {pred} - {cluster_interpretations.get(pred)}")
            if deteksi_ujaran_kebencian(clean_komentar):
                st.error("🚨 Terdeteksi ujaran kebencian!")
            else:
                st.info("Komentar aman.")

            # Plot PCA posisi komentar baru
            pca_new = pca.transform(X_new)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='viridis', ax=ax3, legend=False)
            ax3.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=150)
            ax3.scatter(pca_new[:, 0], pca_new[:, 1], c='black', s=250, marker='P', edgecolor='yellow')
            ax3.set_title("Posisi PCA Komentar Baru")
            st.pyplot(fig3)

    with new_tabs[5]:
        st.subheader("Word Cloud Komentar")
        if 'clean' in df:
            all_words = ' '.join(df['clean'].dropna())
            wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            ax4.imshow(wc, interpolation='bilinear')
            ax4.axis('off')
            st.pyplot(fig4)
        else:
            st.info("Data 'clean' tidak tersedia.")
