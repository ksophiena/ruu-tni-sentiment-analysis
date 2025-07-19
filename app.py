import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

st.set_page_config(page_title="Visualisasi Sentimen RUU TNI", layout="wide")
st.title("Visualisasi Hasil Analisis Sentimen RUU TNI di Media Sosial X")
st.title("📊 Visualisasi Analisis Sentimen terhadap RUU TNI di Media Sosial X")
st.markdown("""
📌 Visualisasi ini bertujuan untuk menyajikan hasil analisis sentimen publik terhadap Rancangan Undang-Undang Tentara Nasional Indonesia (RUU TNI) di media sosial X.

🗓️ Data diambil dari media sosial X pada periode: 
**1 Oktober 2024 – 31 Maret 2025**
""")
st.markdown("---")

try:
    df = pd.read_csv("ruu_tni_sentiment_analysis.csv") 
    st.success("✅Data berhasil dimuat!")

    if 'full_text' in df.columns and 'klasifikasi' in df.columns:

        # Baris 1: WordCloud Seluruh Teks dan WordCloud Positif
        st.subheader("1. WordCloud - Seluruh Teks dan Sentimen Positif")
        col1, col2 = st.columns(2)
        with col1:
            all_text = ' '.join(df['full_text'].astype(str))
            wordcloud_all = WordCloud(
                width=800, height=400, background_color='black',
                stopwords=STOPWORDS, colormap='Blues_r'
            ).generate(all_text)
            st.image(wordcloud_all.to_array(), caption="Seluruh Teks", use_container_width=True)
        with col2:
            text_pos = ' '.join(df[df['klasifikasi'] == 'Positif']['full_text'].astype(str))
            if text_pos.strip():
                wc_pos = WordCloud(
                    width=800, height=400, background_color='white',
                    stopwords=STOPWORDS, colormap='Greens'
                ).generate(text_pos)
                st.image(wc_pos.to_array(), caption="Sentimen Positif", use_container_width=True)
            else:
                st.warning("Tidak ada data untuk sentimen Positif.")

        st.markdown("---")

        # Baris 2: WordCloud Netral dan Negatif
        st.subheader("2. WordCloud - Sentimen Netral dan Negatif")
        col3, col4 = st.columns(2)
        with col3:
            text_netral = ' '.join(df[df['klasifikasi'] == 'Netral']['full_text'].astype(str))
            if text_netral.strip():
                wc_netral = WordCloud(
                    width=800, height=400, background_color='white',
                    stopwords=STOPWORDS, colormap='Blues'
                ).generate(text_netral)
                st.image(wc_netral.to_array(), caption="Sentimen Netral", use_container_width=True)
            else:
                st.warning("Tidak ada data untuk sentimen Netral.")
        with col4:
            text_negatif = ' '.join(df[df['klasifikasi'] == 'Negatif']['full_text'].astype(str))
            if text_negatif.strip():
                wc_negatif = WordCloud(
                    width=800, height=400, background_color='white',
                    stopwords=STOPWORDS, colormap='Reds'
                ).generate(text_negatif)
                st.image(wc_negatif.to_array(), caption="Sentimen Negatif", use_container_width=True)
            else:
                st.warning("Tidak ada data untuk sentimen Negatif.")

        st.markdown("---")

        # Baris 3: Grafik Batang dan Pie Chart
        st.subheader("3. Grafik Distribusi dan Proporsi Sentimen")
        col5, col6 = st.columns(2)
        with col5:
            try:
                bar_img = Image.open("grafik_batang.png")
                st.image(bar_img, caption="Distribusi Sentimen", use_container_width=True)
            except FileNotFoundError:
                st.warning("❗ Gambar 'grafik_batang.png' tidak ditemukan.")
        with col6:
            try:
                pie_img = Image.open("pie_chart.png")
                st.image(pie_img, caption="Proporsi Sentimen", use_container_width=True)
            except FileNotFoundError:
                st.warning("❗ Gambar 'pie_chart.png' tidak ditemukan.")

        st.markdown("---")

        # Baris 4: Confusion Matrix (dikecilkan)
        if 'prediksi' in df.columns:
            st.subheader("4. Confusion Matrix")
            col_cm, _ = st.columns([2, 1])
            with col_cm:
                try:
                    cm_img = Image.open("confusion_matrix.png")
                    st.image(cm_img, caption="Confusion Matrix", use_container_width=True)
                except FileNotFoundError:
                    st.warning("❗ Gambar 'confusion_matrix.png' tidak ditemukan.")
    else:
        st.error("❌ Kolom 'full_text' dan 'klasifikasi' wajib ada di dalam file CSV.")
except FileNotFoundError:
    st.error("❌ File 'ruu_tni_sentiment_analysis.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan program.")
