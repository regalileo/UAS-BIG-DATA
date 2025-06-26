import streamlit as st
import joblib
import re
from PIL import Image

# Load model dan vectorizer
model = joblib.load("model_sentimen.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Styling UI
st.set_page_config(page_title="Klasifikasi Sentimen Komentar", page_icon="💬", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f8;
    }
    .stTextArea textarea {
        background-color: #ffffff;
    }
    .title {
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        color: #333333;
    }
    .result {
        font-size: 1.3em;
        font-weight: bold;
        color: #ffffff;
        background-color: #4caf50;
        padding: 0.5em;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💬 Sistem Klasifikasi Sentimen Komentar</div>', unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center;'>Masukkan komentar Anda untuk mengetahui apakah itu termasuk <strong>kritik</strong>, <strong>pujian</strong>, <strong>netral</strong>, atau <strong>ujaran kebencian</strong>.</p>
""", unsafe_allow_html=True)

# Form Input
komentar = st.text_area("Masukkan komentar di sini:", height=150)

# Fungsi pembersih teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediksi saat tombol diklik
if st.button("🔍 Prediksi Sentimen"):
    if komentar.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        komentar_bersih = clean_text(komentar)
        vektor = vectorizer.transform([komentar_bersih])
        pred = model.predict(vektor)
        label = label_encoder.inverse_transform(pred)[0]

        warna = {
            "kritik": "#fbc02d",
            "pujian": "#4caf50",
            "netral": "#90a4ae",
            "ujaran_kebencian": "#e53935"
        }
        st.markdown(f'<div class="result" style="background-color: {warna.get(label, "#2196f3")};">Hasil Prediksi: {label.upper()}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style='margin-top: 2em;'/>
    <p style='text-align:center; font-size: 0.9em; color: #999999;'>Built with ❤️ using Streamlit & XGBoost</p>
""", unsafe_allow_html=True)