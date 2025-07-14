import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

st.set_page_config(page_title="Visualisasi Sentimen RUU TNI", layout="wide")
st.title("📊 Visualisasi Hasil Analisis Sentimen RUU TNI")
st.markdown("---")

try:
    df = pd.read_csv("ruu_tni_sentiment_analysis.csv") 
    st.success("✅ Data berhasil dimuat!")

    if 'full_text' in df.columns and 'klasifikasi' in df.columns:

        # Seluruh WordCloud
        with st.container():
            st.subheader("1. WordCloud - Seluruh Sentimen")
            all_text = ' '.join(df['full_text'].astype(str))
            wordcloud_all = WordCloud(
                width=2000, height=1000, background_color='black',
                stopwords=STOPWORDS, colormap='Blues_r'
            ).generate(all_text)
            st.image(wordcloud_all.to_array(), use_column_width=True)

        st.markdown("---")

        # WordCloud per Sentimen (horizontal layout)
        st.subheader("2. WordCloud - Klasifikasi Sentimen")
        col1, col2, col3 = st.columns(3)
        sentiments = ['Positif', 'Netral', 'Negatif']
        colors = ['Greens', 'Blues', 'Reds']
        cols = [col1, col2, col3]

        for sentiment, color, col in zip(sentiments, colors, cols):
            with col:
                st.markdown(f"**{sentiment}**")
                text = ' '.join(df[df['klasifikasi'] == sentiment]['full_text'].astype(str))
                if text.strip():
                    wc = WordCloud(width=800, height=400, background_color='white',
                                   stopwords=STOPWORDS, colormap=color).generate(text)
                    st.image(wc.to_array(), use_column_width=True)
                else:
                    st.warning(f"Tidak ada data.")

        st.markdown("---")

        # Grafik Batang
        with st.container():
            st.subheader("3. Grafik Batang - Distribusi Sentimen")
            try:
                bar_img = Image.open("grafik_batang.png")
                st.image(bar_img, caption="Distribusi Sentimen", use_column_width=True)
            except FileNotFoundError:
                st.warning("❗ Gambar 'grafik_batang.png' tidak ditemukan.")

        # Pie Chart
        with st.container():
            st.subheader("4. Pie Chart - Proporsi Sentimen")
            try:
                pie_img = Image.open("pie_chart.png")
                st.image(pie_img, caption="Proporsi Sentimen", use_column_width=True)
            except FileNotFoundError:
                st.warning("❗ Gambar 'pie_chart.png' tidak ditemukan.")

        # Confusion Matrix
        if 'prediksi' in df.columns:
            with st.container():
                st.subheader("5. Confusion Matrix (Gambar)")
                try:
                    cm_img = Image.open("confusion_matrix.png")
                    st.image(cm_img, caption="Confusion Matrix", use_column_width=True)
                except FileNotFoundError:
                    st.warning("❗ Gambar 'confusion_matrix.png' tidak ditemukan.")

    else:
        st.error("❌ Kolom 'full_text' dan 'klasifikasi' wajib ada di dalam file CSV.")
except FileNotFoundError:
    st.error("❌ File 'ruu_tni_sentiment_analysis.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan program.")
