# app.py - Streamlit Clustering Viewer dan Uji Komentar Baru
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from sklearn.decomposition import PCA
from wordcloud import WordCloud

st.set_page_config(page_title="Klasterisasi Komentar Netizen", layout="wide")

# --- Load model dan data ---
vectorizer = joblib.load("vectorizer_tfidf_kmeans.pkl")
model = joblib.load("model_kmeans.pkl")
df = joblib.load("data_clustered.pkl")

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
    url = "https://raw.githubusercontent.com/regalileo/UAS-BIG-DATA/b8349655afe858b4860a4a9b868bbae0b7cf8e89/kata_kasar.txt"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.splitlines()
        return set(k.strip() for k in lines if k.strip())
    else:
        return set()

kata_kasar = load_kata_kasar_from_url()

def deteksi_ujaran_kebencian(teks):
    return any(kata in teks.split() for kata in kata_kasar)

st.markdown("""
    <h2 style='text-align: center;'>Visualisasi & Deteksi Klaster Komentar Netizen</h2>
    <p style='text-align: center;'>Metode TF-IDF dan KMeans Clustering</p>
""", unsafe_allow_html=True)

st.write("---")

# --- Siapkan data untuk visualisasi ---
X = vectorizer.transform(df['clean'])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())
df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]
df['is_hate'] = df['clean'].apply(deteksi_ujaran_kebencian)

# --- Layout 2 kolom utama ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualisasi Klaster dan Distribusi")

    # Plot PCA
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', s=60, ax=ax1)
    ax1.set_title("Pemetaan PCA Klaster Komentar")
    ax1.grid(True)
    st.pyplot(fig1)

    # Donut Chart
    cluster_counts = df['cluster'].value_counts().sort_index()
    labels = [f"Cluster {i}: {n}" for i, n in cluster_counts.items()]
    fig2, ax2 = plt.subplots()
    wedges, texts, autotexts = ax2.pie(cluster_counts, labels=labels,
        autopct="%1.1f%%", startangle=90, wedgeprops={'width':0.5}, textprops={'fontsize': 9})
    ax2.axis("equal")
    st.pyplot(fig2)

with col2:
    st.subheader("Uji Komentar Baru")
    komentar_baru = st.text_area("Masukkan komentar netizen:", height=150)

    if st.button("Prediksi Klaster"):
        if komentar_baru.strip() == "":
            st.warning("Komentar tidak boleh kosong!")
        else:
            clean_komentar = clean_text(komentar_baru)
            vectorized = vectorizer.transform([clean_komentar])
            cluster_pred = model.predict(vectorized)[0]
            st.success(f"Komentar tersebut masuk dalam **Cluster {cluster_pred}**")

            is_hate = deteksi_ujaran_kebencian(clean_komentar)
            if is_hate:
                st.error("Komentar ini terdeteksi mengandung **ujaran kebencian** ⚠️")
            else:
                st.info("Komentar ini **tidak mengandung ujaran kebencian** ✅")
