# app.py - Tampilan Klastering Streamlit
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

# Mengatur konfigurasi halaman untuk tata letak yang lebih lebar
st.set_page_config(page_title="Klasterisasi Komentar Netizen", layout="wide")

# --- Memuat model dan data ---
# Menggunakan st.cache_resource untuk model yang dimuat sekali dan digunakan bersama di seluruh sesi.
@st.cache_resource
def load_model():
    """Memuat vectorizer TF-IDF, model K-Means, dan model PCA yang sudah dilatih."""
    try:
        vectorizer = joblib.load("vectorizer_tfidf_kmeans.pkl")
        model = joblib.load("model_kmeans.pkl")
        pca = joblib.load("pca_model.pkl")
        return vectorizer, model, pca
    except FileNotFoundError:
        # Menampilkan pesan error jika file model tidak ditemukan
        st.error("Error: Pastikan file model 'vectorizer_tfidf_kmeans.pkl', 'model_kmeans.pkl', dan 'pca_model.pkl' ada di direktori yang sama.")
        st.stop() # Menghentikan aplikasi jika model tidak ditemukan
    except Exception as e:
        # Menampilkan pesan error untuk pengecualian lainnya saat memuat model
        st.error(f"Error saat memuat model: {e}")
        st.stop()

# Menggunakan st.cache_data untuk data yang tidak berubah dan memakan waktu untuk dihitung
@st.cache_data
def load_data():
    """Memuat DataFrame yang sudah terklaster."""
    try:
        df = joblib.load("data_clustered.pkl")
        # Memastikan kolom 'cluster' bertipe integer untuk plotting yang konsisten
        df['cluster'] = df['cluster'].astype(int)
        # Memastikan 'pca1' dan 'pca2' ada untuk plotting jika belum ada di df yang dimuat
        # Ini adalah solusi darurat jika komponen PCA tidak disimpan secara eksplisit dengan df
        # Dalam skenario nyata, pca.transform(vectorizer.transform(df['clean'])) akan dilakukan jika PCA tidak disimpan dengan df
        if 'pca1' not in df.columns or 'pca2' not in df.columns:
            st.warning("Kolom 'pca1' atau 'pca2' tidak ditemukan dalam data_clustered.pkl. Melakukan transformasi PCA ulang.")
            # Jika komponen PCA tidak disimpan, lakukan transformasi ulang dari teks bersih
            # Ini memerlukan kolom 'clean' agar tersedia di df
            if 'clean' in df.columns:
                X_transformed = vectorizer.transform(df['clean'])
                df_pca = pca.transform(X_transformed)
                df['pca1'] = df_pca[:, 0]
                df['pca2'] = df_pca[:, 1]
            else:
                st.error("Kolom 'clean' tidak ditemukan di data_clustered.pkl untuk transformasi PCA.")
                st.stop()
        return df
    except FileNotFoundError:
        # Menampilkan pesan error jika file data tidak ditemukan
        st.error("Error: Pastikan file data 'data_clustered.pkl' ada di direktori yang sama.")
        st.stop() # Menghentikan aplikasi jika data tidak ditemukan
    except Exception as e:
        # Menampilkan pesan error untuk pengecualian lainnya saat memuat data
        st.error(f"Error saat memuat data: {e}")
        st.stop()

# Memuat vectorizer, model, dan dataframe setelah fungsi-fungsi didefinisikan
vectorizer, model, pca = load_model()
df = load_data()

# --- Fungsi pembersihan teks (dari notebook Anda) ---
def clean_text(text):
    """
    Melakukan pembersihan teks dasar: huruf kecil, menghapus URL, non-alfanumerik, angka, dan spasi berlebih.
    """
    text = str(text).lower() # Mengubah teks menjadi huruf kecil
    text = re.sub(r"http\S+|www\S+|@\S+", "", text) # Menghapus URL dan mentions
    text = re.sub(r"[^\w\s]", "", text) # Menghapus tanda baca
    text = re.sub(r"\d+", "", text) # Menghapus angka
    text = re.sub(r"\s+", " ", text).strip() # Menghapus spasi berlebih dan spasi di awal/akhir
    return text

# --- Memuat kata kasar dari GitHub ---
@st.cache_data
def load_kata_kasar_from_url():
    """
    Memuat daftar kata-kata kasar dari URL GitHub.
    Mengembalikan set untuk pencarian yang efisien.
    """
    url = "https://raw.githubusercontent.com/regalileo/UAS-BIG-DATA/main/kata_kasar.txt"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Mengaktifkan HTTPError untuk respons yang buruk (4xx atau 5xx)
        lines = response.text.splitlines()
        # Membuat set untuk kompleksitas waktu rata-rata O(1) untuk operasi 'in'
        return set(k.strip() for k in lines if k.strip())
    except requests.exceptions.RequestException as e:
        # Menampilkan peringatan jika gagal memuat kata kasar
        st.warning(f"Gagal memuat daftar kata kasar dari URL: {e}. Deteksi ujaran kebencian mungkin tidak akurat.")
        return set() # Mengembalikan set kosong jika pemuatan gagal
    except Exception as e:
        # Menampilkan peringatan untuk error tak terduga
        st.warning(f"Error tak terduga saat memuat kata kasar: {e}. Deteksi ujaran kebencian mungkin tidak akurat.")
        return set()

# Memuat daftar kata kasar saat aplikasi dimulai
kata_kasar = load_kata_kasar_from_url()

def deteksi_ujaran_kebencian(teks):
    """
    Mendeteksi apakah ada kata dalam teks yang ada di dalam set kata_kasar.
    """
    # Memisahkan teks menjadi kata-kata dan memeriksa apakah ada yang termasuk kata kasar
    return any(kata in teks.split() for kata in kata_kasar)

# --- Interpretasi Klaster (Hardcoded untuk kesederhanaan, bisa dibuat dinamis) ---
# Kamus untuk menyimpan interpretasi nama klaster berdasarkan nomor klaster
cluster_interpretations = {
    0: "Diskusi Umum dan Netral",
    1: "Kritik dan Tuntutan Hukum",
    2: "Dukungan dan Doa Kesehatan untuk Jokowi",
    3: "Fokus pada Najwa Shihab ('Mbak Nana')",
    4: "Diskusi tentang Program 'Mata Najwa'",
    5: "Doa dan Harapan Kesehatan yang Kuat untuk Jokowi",
    6: "Komentar tentang Gerakan Kaki Jokowi (Tremor)"
}

# --- Judul aplikasi ---
st.markdown("""
    <h2 style='text-align: center;'>Visualisasi dan Deteksi Klaster Komentar Media Sosial</h2>
    <p style='text-align: center;'>Aplikasi berbasis TF-IDF + KMeans dengan Streamlit</p>
""", unsafe_allow_html=True) # Mengizinkan penggunaan HTML untuk penataan gaya

st.write("---") # Garis pemisah visual

# --- Tata letak utama menggunakan kolom ---
# col1 untuk visualisasi, col2 untuk input komentar baru dan prediksi
col1, col2 = st.columns([2, 1])

# Spinner untuk menunjukkan bahwa data sedang diproses
with st.spinner("Memproses data untuk visualisasi..."):
    # Mentransformasi teks bersih dari dataframe ke representasi TF-IDF
    X = vectorizer.transform(df['clean'])
    # Mendapatkan posisi centroid dalam ruang asli, lalu mentransformasikannya dengan PCA
    centroids = model.cluster_centers_
    centroids_pca = pca.transform(centroids)

with col1:
    st.subheader("Visualisasi Klaster Dataset")

    # PCA Plot
    st.markdown("### PCA Plot Komentar dan Centroid Klaster")
    # Mengatur gaya plot dan font untuk estetika yang lebih baik
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'Inter'

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    # Scatter plot titik data yang diwarnai berdasarkan klaster
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster',
                    palette='viridis', s=60, ax=ax_pca, alpha=0.7,
                    legend='full') # Menggunakan 'full' untuk menampilkan semua label di legenda
    # Memplot centroid
    ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c='red', s=250, marker='X', label='Centroid Klaster', edgecolor='black', linewidth=1)

    ax_pca.set_title("Visualisasi PCA Klaster Komentar", fontsize=16)
    ax_pca.set_xlabel("Komponen Utama 1", fontsize=12)
    ax_pca.set_ylabel("Komponen Utama 2", fontsize=12)
    ax_pca.grid(True, linestyle='--', alpha=0.6)
    # Menyesuaikan posisi legenda
    ax_pca.legend(title='Klaster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() # Menyesuaikan tata letak untuk mencegah label tumpang tindih
    st.pyplot(fig_pca) # Menampilkan plot PCA di Streamlit

    # Donut Chart Klaster
    st.markdown("### Distribusi Komentar per Klaster")
    # Menghitung jumlah komentar per klaster dan mengurutkannya
    cluster_counts = df['cluster'].value_counts().sort_index()
    # Membuat label untuk donut chart dengan nama klaster dan jumlah
    labels = [f"Klaster {i}: {cluster_interpretations.get(i, 'Tidak Dikenal')} ({n})" for i, n in cluster_counts.items()]

    fig_donut, ax_donut = plt.subplots(figsize=(7, 7))
    # Membuat donut chart
    wedges, texts, autotexts = ax_donut.pie(cluster_counts,
                                            autopct=lambda p: f'{p:.1f}%\n({int(p*sum(cluster_counts)/100)})', # Menampilkan persentase dan jumlah
                                            startangle=90,
                                            wedgeprops={'width': 0.4, 'edgecolor': 'white'},
                                            textprops={'fontsize': 10})
    ax_donut.set_title("Distribusi Komentar per Klaster", fontsize=16)
    ax_donut.axis("equal") # Rasio aspek yang sama memastikan pie digambar sebagai lingkaran.
    # Menambahkan legenda di luar pie untuk keterbacaan yang lebih baik
    ax_donut.legend(wedges, labels,
                    title="Klaster & Jumlah",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    st.pyplot(fig_donut) # Menampilkan donut chart di Streamlit

    # Statistik Deskriptif per Klaster
    st.markdown("### Statistik Deskriptif Dimensi PCA per Klaster")
    # Mengelompokkan data berdasarkan klaster dan menghitung statistik deskriptif untuk komponen PCA
    df_stats = df.groupby('cluster')[['pca1', 'pca2']].agg(['mean', 'std', 'min', 'max']).round(4) # Lebih banyak desimal untuk presisi
    st.dataframe(df_stats.style.set_properties(**{'font-size': '10pt'})) # Menyesuaikan ukuran font untuk dataframe

    # Top Fitur Pembeda (Mutual Info)
    st.markdown("### Top Fitur Pembeda Antar Klaster (Mutual Information)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Mengabaikan potensi peringatan dari mutual_info_classif
        # Mengonversi sparse matrix ke dense array untuk mutual_info_classif jika belum
        if hasattr(vectorizer.transform(df['clean']), 'toarray'):
            X_arr = vectorizer.transform(df['clean']).toarray()
        else:
            X_arr = vectorizer.transform(df['clean']) # Sudah dense
        
        # Menghitung skor Mutual Information
        mi_scores = mutual_info_classif(X_arr, df['cluster'], discrete_features=True)
        top_idx = np.argsort(mi_scores)[::-1][:10] # Mendapatkan 10 indeks teratas
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_idx] # Mendapatkan kata-kata teratas
        top_vals = mi_scores[top_idx] # Mendapatkan nilai skor MI teratas
        top_df = pd.DataFrame({"Fitur": top_words, "Skor MI": top_vals.round(4)}) # Membulatkan skor MI
        st.dataframe(top_df.style.set_properties(**{'font-size': '10pt'})) # Menampilkan dataframe fitur pembeda

with col2:
    st.subheader("Uji Komentar Baru")
    # Area teks untuk input pengguna
    komentar_baru = st.text_area("Masukkan komentar baru di sini:", height=150,
                                 help="Ketik atau tempel komentar YouTube yang ingin Anda klasifikasikan.")

    # Tombol untuk memicu prediksi
    if st.button("Prediksi Klaster", help="Klik untuk mengklasifikasikan komentar yang dimasukkan."):
        if komentar_baru.strip() == "":
            st.warning("Komentar tidak boleh kosong! Harap masukkan teks untuk diproses.")
        else:
            try:
                # Pra-pemrosesan komentar baru
                clean_komentar = clean_text(komentar_baru)
                st.write("✅ Teks berhasil dibersihkan.")

                # Transformasi TF-IDF
                vectorized_new_comment = vectorizer.transform([clean_komentar])
                st.write("✅ Teks berhasil diubah menjadi vektor TF-IDF.")

                # Prediksi Klaster
                cluster_pred = model.predict(vectorized_new_comment)[0]
                # Menampilkan hasil prediksi klaster dengan interpretasinya
                st.success(f"Komentar ini masuk dalam **Klaster {cluster_pred}**: **{cluster_interpretations.get(cluster_pred, 'Interpretasi tidak tersedia')}**")

                # Deteksi Ujaran Kebencian
                is_hate = deteksi_ujaran_kebencian(clean_komentar)
                if is_hate:
                    st.error("🚨 Komentar ini terdeteksi mengandung **ujaran kebencian**. ⚠️")
                else:
                    st.info("👍 Komentar ini **tidak mengandung ujaran kebencian**.")

                # Visualisasi komentar baru pada PCA Plot
                st.markdown("---") # Garis pemisah
                st.subheader("Posisi Komentar Baru pada PCA Plot")
                # Transformasi PCA untuk komentar baru
                new_comment_pca = pca.transform(vectorized_new_comment)

                fig_new_pca, ax_new_pca = plt.subplots(figsize=(10, 6))
                # Scatter plot data klaster yang ada
                sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster',
                                palette='viridis', s=60, ax=ax_new_pca, alpha=0.6, legend=False)
                # Plot centroid klaster yang ada
                ax_new_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                                   c='red', s=250, marker='X', label='Centroid Klaster', edgecolor='black', linewidth=1)
                # Plot komentar baru sebagai titik terpisah
                ax_new_pca.scatter(new_comment_pca[:, 0], new_comment_pca[:, 1],
                                   c='black', s=350, marker='P', label='Komentar Baru', edgecolor='yellow', linewidth=2) # P untuk diprediksi
                ax_new_pca.set_title("PCA Plot: Komentar Baru (Tanda P)", fontsize=16)
                ax_new_pca.set_xlabel("Komponen Utama 1", fontsize=12)
                ax_new_pca.set_ylabel("Komponen Utama 2", fontsize=12)
                ax_new_pca.grid(True, linestyle='--', alpha=0.6)
                ax_new_pca.legend(title='Legenda', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig_new_pca) # Menampilkan plot komentar baru di Streamlit

            except Exception as e:
                # Menangani error saat memproses komentar baru
                st.error("❌ Terjadi error saat memproses komentar.")
                st.exception(e)

# Menampilkan WordCloud dari semua klaster
st.write("---") # Garis pemisah
st.subheader("Word Cloud Keseluruhan Komentar")
# Menggabungkan semua teks bersih menjadi satu string
all_words = ' '.join(df['clean'].dropna())
if all_words:
    # Membuat dan menampilkan WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_words)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off') # Menghilangkan sumbu
    st.pyplot(fig_wc) # Menampilkan word cloud di Streamlit
else:
    st.info("Tidak ada data teks untuk membuat Word Cloud.")
