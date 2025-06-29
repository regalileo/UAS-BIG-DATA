# app.py - Tampilan Klastering Streamlit (Versi Ringkas Fullscreen)
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

# Konfigurasi halaman
st.set_page_config(page_title="Klasterisasi Komentar Netizen", layout="wide")

# CSS untuk padding minimal
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Memuat model
@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer_tfidf_kmeans.pkl")
    model = joblib.load("model_kmeans.pkl")
    pca = joblib.load("pca_model.pkl")
    return vectorizer, model, pca

@st.cache_data
def load_data():
    df = joblib.load("data_clustered.pkl")
    df['cluster'] = df['cluster'].astype(int)
    if 'pca1' not in df.columns:
        vectorizer = joblib.load("vectorizer_tfidf_kmeans.pkl")
        pca = joblib.load("pca_model.pkl")
        X = vectorizer.transform(df['clean'])
        pca_vals = pca.transform(X)
        df['pca1'], df['pca2'] = pca_vals[:, 0], pca_vals[:, 1]
    return df

@st.cache_data
def load_kata_kasar_from_url():
    try:
        url = "https://raw.githubusercontent.com/regalileo/UAS-BIG-DATA/main/kata_kasar.txt"
        res = requests.get(url)
        res.raise_for_status()
        return set(k.strip() for k in res.text.splitlines())
    except:
        return set()

vectorizer, model, pca = load_model()
df = load_data()
kata_kasar = load_kata_kasar_from_url()

cluster_interpretations = {
    0: "Diskusi Umum dan Netral", 1: "Kritik dan Tuntutan Hukum",
    2: "Dukungan dan Doa Kesehatan untuk Jokowi", 3: "Fokus pada Najwa Shihab",
    4: "Diskusi tentang Program 'Mata Najwa'", 5: "Doa dan Harapan untuk Jokowi",
    6: "Komentar tentang Tremor Jokowi"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def deteksi_ujaran_kebencian(teks):
    return any(k in teks.split() for k in kata_kasar)

# Layout tiga kolom
col1, col2, col3 = st.columns([1.2, 1.2, 1])

with col1:
    st.subheader("Distribusi Klaster")
    cluster_counts = df['cluster'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    wedges, _, autotexts = ax1.pie(cluster_counts, autopct=lambda p: f'{p:.1f}%\n({int(p*sum(cluster_counts)/100)})',
                                   startangle=90, wedgeprops={'width':0.4}, textprops={'fontsize':9})
    ax1.axis('equal')
    st.pyplot(fig1)
    st.dataframe(pd.DataFrame({
        'Klaster': [f"Klaster {i}" for i in cluster_counts.index],
        'Interpretasi': [cluster_interpretations.get(i, '-') for i in cluster_counts.index],
        'Jumlah': cluster_counts.values
    }), hide_index=True)

with col2:
    st.subheader("PCA dan WordCloud")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    centroids_pca = pca.transform(model.cluster_centers_)
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='viridis', ax=ax2, legend=None)
    ax2.scatter(centroids_pca[:,0], centroids_pca[:,1], c='red', s=200, marker='X', edgecolor='black')
    ax2.set_title("PCA Komentar")
    st.pyplot(fig2)

    words = ' '.join(df['clean'].dropna())
    if words:
        wc = WordCloud(width=400, height=200).generate(words)
        fig_wc, ax_wc = plt.subplots(figsize=(5,2.5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

with col3:
    st.subheader("Uji Komentar Baru")
    komentar = st.text_area("Komentar:", height=100)
    if st.button("Prediksi"):
        if komentar.strip():
            teks_bersih = clean_text(komentar)
            vektor = vectorizer.transform([teks_bersih])
            pred = model.predict(vektor)[0]
            st.success(f"Masuk Klaster {pred}: {cluster_interpretations.get(pred)}")
            if deteksi_ujaran_kebencian(teks_bersih):
                st.error("Terdeteksi ujaran kebencian.")
            else:
                st.info("Tidak mengandung ujaran kebencian.")
            titik = pca.transform(vektor)
            fig3, ax3 = plt.subplots(figsize=(5,3))
            sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='viridis', ax=ax3, legend=None)
            ax3.scatter(centroids_pca[:,0], centroids_pca[:,1], c='red', s=150, marker='X')
            ax3.scatter(titik[:,0], titik[:,1], c='black', s=200, marker='P', edgecolor='yellow')
            ax3.set_title("Posisi Komentar Baru")
            st.pyplot(fig3)
        else:
            st.warning("Komentar tidak boleh kosong.")

# Tab Tambahan
st.write("---")
tab1, tab2 = st.tabs(["📊 Statistik PCA", "🔍 Fitur Pembeda"])

with tab1:
    st.dataframe(df.groupby('cluster')[['pca1','pca2']].agg(['mean','std','min','max']).round(3))

with tab2:
    X_arr = vectorizer.transform(df['clean']).toarray()
    mi = mutual_info_classif(X_arr, df['cluster'])
    idx_top = np.argsort(mi)[::-1][:10]
    top_words = vectorizer.get_feature_names_out()[idx_top]
    st.dataframe(pd.DataFrame({'Fitur': top_words, 'Skor MI': mi[idx_top].round(4)}))
