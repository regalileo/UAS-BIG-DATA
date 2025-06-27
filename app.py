# app.py - Streamlit Clustering Viewer dan Uji Komentar Baru
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
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

# --- Kata kasar ---
kata_kasar = {
    'tolol', 'anjing', 'mampus', 'bangsat', 'goblok', 'tai', 'kontol', 'bajingan', 'brengsek',
    'keparat', 'kampret', 'idiot', 'bego', 'sinting', 'gila', 'laknat', 'brengsek', 'fuck', 'fucking',
    'shit', 'bitch', 'kafir', 'pengkhianat', 'komunis', 'aseng', 'cina', 'penjilat', 'pantek', 'setan'
}

def deteksi_ujaran_kebencian(teks):
    return any(kata in teks.split() for kata in kata_kasar)

st.markdown("""
    <h2 style='text-align: center;'>Visualisasi & Deteksi Klaster Komentar Netizen</h2>
    <p style='text-align: center;'>Metode TF-IDF dan KMeans Clustering</p>
""", unsafe_allow_html=True)

st.write("---")

# --- PCA Visualisasi ---
st.subheader("Visualisasi Klaster Komentar (PCA)")
X = vectorizer.transform(df['clean'])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())
df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]

fig1, ax1 = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', s=60, ax=ax1)
ax1.set_title("Pemetaan PCA Klaster Komentar")
ax1.grid(True)
st.pyplot(fig1)

# --- Donut Chart ---
st.subheader("Distribusi Jumlah Komentar per Klaster")
cluster_counts = df['cluster'].value_counts().sort_index()
fig2, ax2 = plt.subplots()
ax2.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index],
        autopct="%1.1f%%", startangle=90, wedgeprops={'width':0.5})
ax2.axis("equal")
st.pyplot(fig2)

# --- WordCloud Komentar Kasar ---
st.subheader("WordCloud Komentar Mengandung Ujaran Kebencian")
df['is_hate'] = df['clean'].apply(deteksi_ujaran_kebencian)
text_kasar = " ".join(df[df['is_hate']]['clean'])
if text_kasar:
    wc_fig, wc_ax = plt.subplots(figsize=(10, 5))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_kasar)
    wc_ax.imshow(wordcloud, interpolation='bilinear')
    wc_ax.axis("off")
    st.pyplot(wc_fig)
else:
    st.info("Tidak ditemukan komentar yang mengandung ujaran kebencian pada dataset.")

# --- Input komentar baru ---
st.write("---")
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
