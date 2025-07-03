import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

st.set_page_config(page_title="Visualisasi Sentimen RUU TNI", layout="wide")
st.title("📊 Visualisasi Hasil Analisis Sentimen RUU TNI")

try:
    df = pd.read_csv("ruu_tni_sentiment_analysis.csv") 
    st.success("✅ Data berhasil dimuat!")

    if 'full_text' in df.columns and 'klasifikasi' in df.columns:
        # WordCloud Seluruh Teks
        st.subheader("1. WordCloud - Seluruh Sentimen")
        all_text = ' '.join(df['full_text'].astype(str))
        wordcloud_all = WordCloud(
            width=2000, height=1000, background_color='black',
            stopwords=STOPWORDS, colormap='Blues_r'
        ).generate(all_text)
        st.image(wordcloud_all.to_array())

        # WordCloud per Sentimen
        st.subheader("2. WordCloud - Klasifikasi Sentimen")
        for sentiment, color in zip(['Positif', 'Netral', 'Negatif'], ['Greens', 'Blues', 'Reds']):
            st.markdown(f"**{sentiment}**")
            text = ' '.join(df[df['klasifikasi'] == sentiment]['full_text'].astype(str))
            if text.strip():
                wc = WordCloud(width=2000, height=1000, background_color='white',
                               stopwords=STOPWORDS, colormap=color).generate(text)
                st.image(wc.to_array())
            else:
                st.warning(f"Tidak ada data untuk sentimen {sentiment}.")

        # Grafik Batang 
        st.subheader("3. Grafik Batang - Distribusi Sentimen")
        try:
            bar_img = Image.open("grafik_batang.png")
            st.image(bar_img, caption="Distribusi Sentimen", use_container_width=True)
        except FileNotFoundError:
            st.warning("❗ Gambar 'grafik_batang.png' tidak ditemukan.")

        # Pie Chart 
        st.subheader("4. Pie Chart - Proporsi Sentimen")
        try:
            pie_img = Image.open("pie_chart.png")
            st.image(pie_img, caption="Proporsi Sentimen", use_container_width=True)
        except FileNotFoundError:
            st.warning("❗ Gambar 'pie_chart.png' tidak ditemukan.")

        # Confusion Matrix 
        if 'prediksi' in df.columns:
            st.subheader("5. Confusion Matrix (Gambar)")
            try:
                cm_img = Image.open("confusion_matrix.png")
                st.image(cm_img, caption="Confusion Matrix", use_container_width=True)
            except FileNotFoundError:
                st.warning("❗ Gambar 'confusion_matrix.png' tidak ditemukan.")
    else:
        st.error("❌ Kolom 'full_text' dan 'klasifikasi' wajib ada di dalam file CSV.")
except FileNotFoundError:
    st.error("❌ File 'ruu_tni_sentiment_analysis.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan program.")
