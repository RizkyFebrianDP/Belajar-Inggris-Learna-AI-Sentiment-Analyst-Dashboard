#import library
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# load dataset
data_clean = pd.read_csv("data/cleaned_dataset.csv")
data = data_clean.copy()
data = data.drop_duplicates(subset='final')
print(f'nilai duplicate adalah =',data.duplicated().sum())

# Inisialisasi SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
data['final'] =  data['final'].astype(str)

# Create a function to calculate sentiment scores
def sentiment(text):
    te = sia.polarity_scores(text)
    return te

# Applying the function
data['sentiment_scores'] = data['final'].apply(sentiment) 
data['sentiment'] = data['sentiment_scores'].apply(
    lambda x: "POSITIVE" if x['compound'] >= 0.05 else "NEGATIVE"
)
label = data[['final','sentiment']]

#Raw Labeling_data
label.to_csv("raw_labeling_data.csv")
sentiment_counts = label['sentiment'].value_counts()

# Menghitung proporsi dari setiap kategori sentimen
sentiment_proportions = label['sentiment'].value_counts(normalize=True)

# Printing the results
print("Jumlah sentimen:")
print(sentiment_counts)
print("\nProporsi sentimen:")
print(sentiment_proportions)

# Counting the number of each sentiment category
sentiment_counts = label['sentiment'].value_counts()
print("Jumlah sentimen:")
print(sentiment_counts)

# Undersampling for class imbalance, in this case, we will undersample the majority class (POSITIVE)
positif = label[label['sentiment'] == 'POSITIVE']
negatif = label[label['sentiment'] == 'NEGATIVE']

n_negatif = len(negatif)
positif_undersampled = positif.sample(n=n_negatif, random_state=42)
data_balanced = pd.concat([positif_undersampled, negatif])

# Menghitung jumlah dan proporsi setelah undersampling
sentiment_counts_balanced = data_balanced['sentiment'].value_counts()
sentiment_proportions_balanced = data_balanced['sentiment'].value_counts(normalize=True)

#Printing the results after undersampling
print("\nJumlah sentimen setelah undersampling:")
print(sentiment_counts_balanced)

print("\nProporsi sentimen setelah undersampling:")
print(sentiment_proportions_balanced)

# save the balanced data to a new CSV file
data_balanced.to_csv("labeling_data.csv", index=False)