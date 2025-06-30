
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Visualisasi Sentimen RUU TNI", layout="wide")
st.title("📊 Visualisasi Sentimen RUU TNI dari File CSV")

df = pd.read_csv("ruu_tni_sentiment_analysis.csv")

    if 'full_text' in df.columns and 'klasifikasi' in df.columns:
        st.success("✅ Data berhasil dimuat!")

        # WordCloud Seluruh Teks
        st.subheader("1. WordCloud - Seluruh Teks")
        all_text = ' '.join(df['full_text'].astype(str))
        wordcloud_all = WordCloud(
            width=2000, height=1000, background_color='black',
            stopwords=STOPWORDS, colormap='Blues_r'
        ).generate(all_text)
        st.image(wordcloud_all.to_array())

        # WordCloud per Sentimen
        st.subheader("2. WordCloud per Sentimen")
        for sentiment, color in zip(['Positif', 'Netral', 'Negatif'], ['Greens', 'Blues', 'Reds']):
            st.markdown(f"**{sentiment}**")
            text = ' '.join(df[df['klasifikasi'] == sentiment]['full_text'].astype(str))
            if text.strip():
                wc = WordCloud(width=2000, height=1000, background_color='white',
                               stopwords=STOPWORDS, colormap=color).generate(text)
                st.image(wc.to_array())
            else:
                st.warning(f"Tidak ada data untuk sentimen {sentiment}.")

        # Grafik Bar
        st.subheader("3. Grafik Bar - Distribusi Sentimen")
        sentiment_counts = df['klasifikasi'].value_counts()
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                    palette=['#5CB338', '#2394f7', '#f72323'], ax=ax_bar)
        ax_bar.set_xlabel("Sentimen")
        ax_bar.set_ylabel("Jumlah")
        ax_bar.set_title("Distribusi Sentimen")
        for i, v in enumerate(sentiment_counts.values):
            ax_bar.text(i, v + 0.5, str(v), ha='center')
        st.pyplot(fig_bar)

        # Pie Chart
        st.subheader("4. Pie Chart - Proporsi Sentimen")
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(sentiment_counts.values, labels=sentiment_counts.index,
                   autopct='%1.1f%%', colors=['#BFF6C3', '#C6E7FF', '#F7CFD8'],
                   startangle=90, textprops={'fontsize': 12})
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

        # Confusion Matrix (jika ada kolom 'prediksi')
        if 'prediksi' in df.columns:
            st.subheader("5. Confusion Matrix")
            cm = confusion_matrix(df['klasifikasi'], df['prediksi'],
                                  labels=["Positif", "Netral", "Negatif"])
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["Positif", "Netral", "Negatif"],
                        yticklabels=["Positif", "Netral", "Negatif"],
                        ax=ax_cm)
            ax_cm.set_xlabel("Prediksi")
            ax_cm.set_ylabel("Aktual")
            st.pyplot(fig_cm)

            st.subheader("Classification Report")
            report = classification_report(df['klasifikasi'], df['prediksi'], digits=4)
            st.code(report)
        else:
            st.info("❗ Kolom 'prediksi' tidak ditemukan. Hanya menampilkan label aktual.")
    else:
        st.error("❌ Kolom 'full_text' dan 'klasifikasi' wajib ada di dalam file CSV.")
