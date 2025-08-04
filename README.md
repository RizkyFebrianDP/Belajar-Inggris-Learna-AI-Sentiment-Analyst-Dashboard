<<<<<<< HEAD
# Belajar-Inggris-Learna-AI-Sentiment-Analyst-Dashboard

## Project Overview
This project is a Sentiment Analysis Dashboard for the "Belajar Inggris: Learna AI" app, which is an AI-powered English learning application available on the Google Play Store. The dashboard analyzes user comments and feedback from the app, classifying them into positive or negative sentiments using various machine learning models. The goal is to provide insights into user opinions and improve the app experience based on sentiment trends.

## Code Flow and Components

### Data Scraping (`scrap.py`)
- Scrapes user reviews from the Google Play Store for the "Belajar Inggris: Learna AI" app.
- Saves the raw comments into a CSV file (`raw_dataset.csv`).

### Data Preprocessing (`src/preprocessing/preprocessing.py`)
- Cleans the raw text data by removing emojis, URLs, special characters, and stopwords.
- Performs normalization, lemmatization, and tokenization.
- Saves the cleaned dataset for further processing.

### Sentiment Labeling (`src/preprocessing/labeling_ml.py`)
- Uses the VADER sentiment analyzer to assign sentiment labels (POSITIVE or NEGATIVE) to the cleaned text.
- Balances the dataset by undersampling the majority class.
- Saves the labeled dataset (`labeling_data.csv`) for model training.

### Model Training (`src/models/traditional_ml/`)
- Implements three traditional machine learning models:
  - Random Forest (`random_forest.py`)
  - Naive Bayes (`naive_bayes.py`)
  - Logistic Regression (`logistic_regression.py`)
- Each model:
  - Loads the labeled data.
  - Transforms text data into TF-IDF features.
  - Trains the model and evaluates accuracy.
  - Saves the trained model and vectorizer for later use.

### Dashboard (`src/dashboard/dashboard.py`)
- Built with Streamlit to provide an interactive web interface.
- Visualizes:
  - Most frequent words in user comments.
  - Sentiment distribution pie chart.
  - Model performance comparison (accuracy, precision, recall, F1 score).
- Allows users to input text and get sentiment predictions from selected models.

## Closing
This project demonstrates an end-to-end pipeline for sentiment analysis on app reviews, from data collection to model deployment and visualization. It can be extended with more advanced models or additional data sources. For any questions or contributions, please feel free to contact the project maintainer.

## Installation
To install all required dependencies, please use the provided requirements file. Run the following command in your terminal:

```
pip install -r https://raw.githubusercontent.com/RizkyFebrianDP/Belajar-Inggris-Learna-AI-Sentiment-Analyst-Dashboard/main/requirements.txt
```

