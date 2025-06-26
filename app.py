import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from deep_translator import GoogleTranslator
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Analisis Sentimen RUU TNI", layout="wide")
st.title("📊 Analisis Sentimen Twitter terhadap RUU TNI")

# Upload file
uploaded_file = st.file_uploader("📁 Upload file CSV berisi data Twitter", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df[['full_text', 'username', 'created_at']].dropna().drop_duplicates()
    st.success("✅ Data berhasil dimuat!")

    st.subheader("1. Preprocessing Data")

    # Case folding & cleaning
    norm_dict = {
        "yg": "yang", "kaya": "seperti", "bgt": "banget", "engga": "tidak", "gak": "tidak",
        "pd": "pada", "kl": "kalau", "jgn": "jangan", "lg": "lagi", "sampe": "sampai", "blm": "belum",
        "makin": "semakin", "dgn": "dengan", "org": "orang", "udh": "sudah", "sma": "sama",
        "krn": "karena", "kyk": "seperti", "dr": "dari", "hrs": "harus", "ttep": "tetap"
    }

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', ' ', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        for word, replacement in norm_dict.items():
            text = text.replace(word, replacement)
        return text

    df['cleaned'] = df['full_text'].apply(clean_text)

    # Stopword Removal & Stemming
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words() + ['tidak']
    dictionary = ArrayDictionary(stopwords)
    remover = factory.create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()

    df['cleaned'] = df['cleaned'].apply(lambda x: remover.remove(x))
    df['stemmed'] = df['cleaned'].apply(lambda x: " ".join([stemmer.stem(w) for w in x.split()]))

    # Translate
    st.info("🔁 Menerjemahkan teks (bisa lambat, tergantung jumlah data)...")
    def translate(text):
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except:
            return text
    df['translated'] = df['stemmed'].apply(translate)

    # Sentiment Analysis
    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        return 'Positif' if polarity > 0 else 'Negatif' if polarity < 0 else 'Netral'

    df['sentimen'] = df['translated'].apply(get_sentiment)

    # Distribusi Sentimen
    st.subheader("2. Distribusi Sentimen")
    sentiment_counts = df['sentimen'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel', ax=ax)
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Sentimen")
    st.pyplot(fig)

    # Wordcloud
    st.subheader("3. Wordcloud per Sentimen")
    for s in ['Positif', 'Netral', 'Negatif']:
        st.markdown(f"**{s}**")
        text = ' '.join(df[df['sentimen'] == s]['cleaned'])
        if text:
            wc = WordCloud(width=1000, height=500, background_color='white').generate(text)
            st.image(wc.to_array())
        else:
            st.warning("Tidak ada data untuk ditampilkan.")

    # Modeling Naive Bayes
    st.subheader("4. Evaluasi Model Naive Bayes")
    X = df['cleaned']
    y = df['sentimen']
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vec, y)
    y_pred = model.predict(X_vec)

    cm = confusion_matrix(y, y_pred, labels=["Positif", "Netral", "Negatif"])
    report = classification_report(y, y_pred, digits=4, output_dict=False)
    st.code(report)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Positif", "Netral", "Negatif"], yticklabels=["Positif", "Netral", "Negatif"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)
