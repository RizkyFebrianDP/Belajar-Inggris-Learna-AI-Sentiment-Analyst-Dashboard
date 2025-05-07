import streamlit as st
import joblib
from transformers import pipeline
import os
import pandas as pd
import numpy as np
from google_play_scraper import reviews, Sort
import sys
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.preprocessing import clean, casefoldingText, lemmas, token_stopword, toSentence
from src.preprocessing.labeling_ml import sentiment

# Load saved models and vectorizer
model_folder = 'src/models/save_models'
random_forest_model = joblib.load(os.path.join(model_folder, 'random_forest_model.pkl'))
logistic_regression_model = joblib.load(os.path.join(model_folder, 'logistic_regression_model.pkl'))
naive_bayes_model = joblib.load(os.path.join(model_folder, 'naive_bayes_model.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(model_folder, 'tfidf_vectorizer.pkl'))

# Initialize DistilBERT pipeline
distilbert_classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

def predict_traditional_models(text):
    # Transform text using the loaded vectorizer
    text_tfidf = tfidf_vectorizer.transform([text]).toarray()
    # Predict with each traditional model
    rf_pred = random_forest_model.predict(text_tfidf)[0]
    lr_pred = logistic_regression_model.predict(text_tfidf)[0]
    nb_pred = naive_bayes_model.predict(text_tfidf)[0]
    return rf_pred, lr_pred, nb_pred

def predict_deep_learning_model(text):
    result = distilbert_classifier(text)
    return result[0]['label']

def scrape_playstore_reviews(app_id, count=1000):
    result, _ = reviews(
        app_id,
        lang='en',
        country='us',
        sort=Sort.MOST_RELEVANT,
        count=count,
        filter_score_with=None
    )
    df = pd.DataFrame(np.array(result), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    comments = df['content'].astype(str).tolist()
    return comments

def preprocess_comments(comments):
    cleaned = [toSentence(token_stopword(lemmas(casefoldingText(clean(c))))) for c in comments]
    return cleaned

def analyze_sentiments(comments):
    sentiments = [sentiment(c) for c in comments]
    labels = ['POSITIVE' if s['compound'] >= 0.05 else 'NEGATIVE' for s in sentiments]
    return labels

def plot_pie_chart(data):
    fig, ax = plt.subplots()
    ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999','#99ff99'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

def plot_length_histogram(comments, labels):
    lengths = [len(c.split()) for c in comments]
    df = pd.DataFrame({'length': lengths, 'sentiment': labels})
    fig, ax = plt.subplots()
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        ax.hist(subset['length'], bins=20, alpha=0.5, label=sentiment)
    ax.set_xlabel('Comment Length (words)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Comment Lengths by Sentiment')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Play Store Comments Sentiment Analysis Dashboard")
    st.write("Enter a Play Store app ID or link to scrape comments and analyze sentiment.")

    app_input = st.text_input("Play Store App ID or URL", "")

    model_options = {
        "Random Forest": lambda text: predict_traditional_models(text)[0],
        "Logistic Regression": lambda text: predict_traditional_models(text)[1],
        "Naive Bayes": lambda text: predict_traditional_models(text)[2],
        "DistilBERT (Deep Learning)": predict_deep_learning_model,
    }

    selected_model = st.selectbox("Select Model for Prediction", list(model_options.keys()))
    
    if st.button("Analyze"):
        if not app_input.strip():
            st.warning("Please enter a Play Store app ID or URL.")
            return

        # Extract app ID from URL if full URL is given
        if 'play.google.com' in app_input:
            import re
            match = re.search(r'id=([a-zA-Z0-9._]+)', app_input)
            if match:
                app_id = match.group(1)
            else:
                st.error("Invalid Play Store URL. Please enter a valid app ID or URL.")
                return
        else:
            app_id = app_input.strip()

        with st.spinner("Scraping Play Store reviews..."):
            comments = scrape_playstore_reviews(app_id, count=1000)

        if not comments:
            st.error("No comments found for this app.")
            return

        with st.spinner("Preprocessing comments..."):
            preprocessed_comments = preprocess_comments(comments)

        with st.spinner("Predicting sentiments..."):
            predictions = []
            if selected_model == "GPT API":
                for c in preprocessed_comments:
                    pred = model_options[selected_model](c)
                    predictions.append(pred)
            else:
                predictions = [model_options[selected_model](c) for c in preprocessed_comments]

        # Aggregate prediction counts
        prediction_counts = pd.Series(predictions).value_counts()

        st.subheader("Sentiment")
        st.bar_chart(prediction_counts)
        st.write("Total:")
        st.write(prediction_counts)

        st.subheader("Sentiment Distribution Pie Chart")
        plot_pie_chart(prediction_counts)


        # Additional visualizations
        st.subheader("Word Cloud for Positive Comments")
        positive_comments = [preprocessed_comments[i] for i, label in enumerate(predictions) if label == 'POSITIVE']
        if positive_comments:
            plot_wordcloud(positive_comments, "Positive Comments Word Cloud")
        else:
            st.write("No positive comments to display word cloud.")

        st.subheader("Word Cloud for Negative Comments")
        negative_comments = [preprocessed_comments[i] for i, label in enumerate(predictions) if label == 'NEGATIVE']
        if negative_comments:
            plot_wordcloud(negative_comments, "Negative Comments Word Cloud")
        else:
            st.write("No negative comments to display word cloud.")

        st.subheader("Histogram of Comment Lengths by Sentiment")
        plot_length_histogram(preprocessed_comments, predictions)

if __name__ == "__main__":
    main()
