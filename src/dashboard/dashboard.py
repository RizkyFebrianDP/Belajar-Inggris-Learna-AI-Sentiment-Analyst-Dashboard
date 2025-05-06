# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Instructions for installing required packages
st.sidebar.header("Setup Instructions")
st.sidebar.markdown("""
If you don't have the required packages installed, run the following command in your terminal:

```
<<<<<<< 
pip install -r https://raw.githubusercontent.com/RizkyFebrianDP/Belajar-Inggris-Learna-AI-Sentiment-Analyst-Dashboard/main/requirements.txt
```
""")



@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw_labeling_data.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/raw_labeling_data.csv')
        except FileNotFoundError:
            st.error("Labeled data file not found. Please place 'labeling_data.csv' in the 'data/' folder or current directory.")
            return None
    return df

#wordcloud visualization
@st.cache_data
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def plot_wordcloud(wordcloud):
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud.to_array(), interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

#frequent words visualization
def plot_most_frequent_words(text, top_n=20):
    words = text.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)
    words, counts = zip(*most_common)

    plt.figure(figsize=(12,6))
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    st.pyplot(plt)

#sentiment pie chart visualization
def plot_sentiment_pie(df):
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                 title='Proportion of Positive and Negative Sentences',
                 color=sentiment_counts.index,
                 color_discrete_map={'POSITIVE':'green', 'NEGATIVE':'red'})
    
    fig.update_layout(
        title_font_size=20,
    )
    st.plotly_chart(fig)

#model performance visualization
def evaluate_model(model_name, model_path, vectorizer_path=None):
    df = pd.read_csv('data/labeling_data.csv')
    df.dropna(subset=['final', 'sentiment'], inplace=True)
    X = df['final']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    if vectorizer_path:
        vectorizer = joblib.load(vectorizer_path)
        X_test_transformed = vectorizer.transform(X_test)
        X_test_array = X_test_transformed.toarray()

    model = joblib.load(model_path)

    if vectorizer_path:
        y_pred = model.predict(X_test_array)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='POSITIVE', average='binary')
    recall = recall_score(y_test, y_pred, pos_label='POSITIVE', average='binary')
    f1 = f1_score(y_test, y_pred, pos_label='POSITIVE', average='binary')
    return accuracy, precision, recall, f1

def plot_model_performance():
    st.subheader("Model Performance Comparison")

    models_info = {
        'Random Forest': {
            'model_path': 'src/models/save_models/random_forest_model.pkl',
            'vectorizer_path': 'src/models/save_models/tfidf_vectorizer.pkl'
        },
        'Naive Bayes': {
            'model_path': 'src/models/save_models/naive_bayes_model.pkl',
            'vectorizer_path': 'src/models/save_models/tfidf_vectorizer.pkl'
        },
        'Logistic Regression': {
            'model_path': 'src/models/save_models/logistic_regression_model.pkl',
            'vectorizer_path': 'src/models/save_models/tfidf_vectorizer.pkl'
        }
    }

    performance_data = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    for model_name, paths in models_info.items():
        try:
            accuracy, precision, recall, f1 = evaluate_model(
                model_name,
                paths['model_path'],
                paths['vectorizer_path']
            )
        except Exception as e:
            st.warning(f"Could not evaluate {model_name}: {e}")
            accuracy = precision = recall = f1 = 0.0

        performance_data['Model'].append(model_name)
        performance_data['Accuracy'].append(accuracy)
        performance_data['Precision'].append(precision)
        performance_data['Recall'].append(recall)
        performance_data['F1 Score'].append(f1)

    perf_df = pd.DataFrame(performance_data)
    # Format only numeric columns
    st.dataframe(perf_df.style.format({col: "{:.2f}" for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']}))


def simulate_prediction(text, model_name):
    if len(text.strip()) == 0:
        return "Please enter some text for prediction."
    if len(text) % 2 == 0:
        pred = "Positive"
    else:
        pred = "Negative"
    return f"Model '{model_name}' predicts: {pred}"

def main():
    st.title("Belajar Inggris: Learna AI Sentiment Analysis Dashboard")
    df = load_data()
    if df is None:
        return

    # Create containers to separate wordcloud and prediction UI
    wordcloud_container = st.container()
    prediction_container = st.container()

    # Render wordcloud and other static visualizations in wordcloud_container
    with wordcloud_container:
        st.header("Description of Project")
        st.write("This project is a sentiment analysis dashboard for analyzing user comments from the Belajar Inggris: Learna AI app on the Google Play Store, using machine learning models to classify feedback as positive or negative and visualize data and model performance. Developed by Codeway Dijital, the app is an AI-powered English learning tool offering interactive lessons, personalized grammar, vocabulary, and pronunciation exercises . "
                 "It uses various machine learning models to classify comments as positive or negative. "
                 "The dashboard allows users to visualize the data and model performance.")
        if 'final' in df.columns:
            all_text = " ".join(df['final'].astype(str).tolist())
            st.subheader("Most Frequent Words")
            cached_wordcloud = generate_wordcloud(all_text)
            plot_wordcloud(cached_wordcloud)
            st.subheader("Most Frequent Words - Bar Chart")
            plot_most_frequent_words(all_text)
        else:
            st.warning("Column 'final' not found in data for word cloud visualization.")


        if 'sentiment' in df.columns:
            plot_sentiment_pie(df)
        else:
            st.warning("Column 'sentiment' not found in data for sentiment proportion visualization.")
        plot_model_performance()

    # Prediction UI in separate container
    with prediction_container:
        st.subheader("Prediction with Selected Model")
        user_text = st.text_area("Enter text to predict sentiment:")
        model_options = ['Random Forest', 'Naive Bayes', 'Logistic Regression', 'DistilBERT']
        selected_model = st.selectbox("Select Model", model_options)

        if st.button("Predict"):
            result = simulate_prediction(user_text, selected_model)
            if "positive" in result.lower():
                st.success(result)
            elif "negative" in result.lower():
                st.error(result)
            else:
                st.info(result)

if __name__ == "__main__":
    main()
