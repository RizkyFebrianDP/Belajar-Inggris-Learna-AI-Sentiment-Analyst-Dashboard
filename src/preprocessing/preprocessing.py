#import libraries
import pandas as pd
import re
import emoji
import contractions
import spacy
import requests

#Load Dataset
raw_data = pd.read_csv("data/raw_dataset.csv")
null = raw_data.isna().sum().sum()
duplicates =raw_data.duplicated().sum().sum()

#Check Data
print(f'''
null value:{null}
duplicate value:{duplicates}
''')

#Data Cleaning
def clean(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r'@[\S.+=]+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'https[\S\s\.]+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'#[\w]+', '', text)
    text = re.sub(r'"', '', text)
    text = re.sub(r'TXT[\w]+', '', text)
    text = " ".join(text.split())
    text = text.strip()

    return text

def casefoldingText(text):
  text = text.lower()
  return text

cleaned = raw_data['content'].apply(clean).apply(casefoldingText)
cleaned = cleaned.drop_duplicates()
cleaned = cleaned.loc[cleaned.str.strip() != ""]
cleaned_df = cleaned.to_frame(name='content')


nlp = spacy.load("en_core_web_sm")
def lemmas(text):
  docs = nlp(text)
  return ' '.join([token.lemma_ for token in docs])

#Normalize and Lematization
cleaned_df['norm'] = cleaned_df['content'].apply(contractions.fix)
cleaned_df['norm'] = cleaned_df['norm'].apply(lemmas)

#Tokenize and StopWord
url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
response = requests.get(url)
stopwords_list = response.text.splitlines()

for word in stopwords_list:
    nlp.Defaults.stop_words.add(word)

symbol = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+',
           '[', ']', '{', '}', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/', '~', '`',' .']

for word2 in symbol:
   nlp.Defaults.stop_words.add(word2)

def token_stopword(text):
      doc = nlp(text)
      return [token.text for token in doc if not token.is_stop and not token.is_punct]


#Token to Sentence
def toSentence(list_words):
    sentence = ' '.join(word for word in list_words)
    return sentence

cleaned_df['tokens'] = cleaned_df['norm'].apply(token_stopword)
cleaned_df['final'] = cleaned_df['tokens'].apply(toSentence)
print(cleaned_df['final'].isna().sum().sum())

#save the cleaned dataset
cleaned_df.to_csv("cleaned_dataset.csv")