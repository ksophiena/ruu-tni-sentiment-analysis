import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

st.set_page_config(page_title="Visualisasi Sentimen RUU TNI", layout="wide")
st.title("Visualisasi Hasil Analisis Sentimen RUU TNI")
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

        st.markdown("## 📊 Distribusi Sentimen (Prediksi)")
        sentiment_counts = df['prediksi'].value_counts()
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax_bar)
        ax_bar.set_title("Jumlah Tweet per Sentimen (Prediksi)")
        ax_bar.set_ylabel("Jumlah")
        ax_bar.set_xlabel("Sentimen")
        st.pyplot(fig_bar)
        
        st.markdown("## 🧩 Diagram Lingkaran Proporsi Sentimen (Prediksi)")
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax_pie.axis("equal")
        st.pyplot(fig_pie)
        
        st.markdown("## 🧪 Confusion Matrix dan Evaluasi Model")
        true_labels = df['klasifikasi']
        pred_labels = df['prediksi']
        cm = confusion_matrix(true_labels, pred_labels, labels=["Positif", "Netral", "Negatif"])
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Positif", "Netral", "Negatif"],
                    yticklabels=["Positif", "Netral", "Negatif"],
                    ax=ax_cm)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Label Asli")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        
        st.markdown("### 📈 Classification Report")
        report = classification_report(true_labels, pred_labels, target_names=["Positif", "Netral", "Negatif"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

    else:
        st.error("❌ Kolom 'full_text' dan 'klasifikasi' wajib ada di dalam file CSV.")
except FileNotFoundError:
    st.error("❌ File 'ruu_tni_sentiment_analysis.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan program.")
