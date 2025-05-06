#Load Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the data
df = pd.read_csv("data/labeling_data.csv")
df.dropna(subset=['final', 'sentiment'], inplace=True)

#Split the data into features and target variable
X = df['final']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

#transform the text data into TF-IDF features
tfidf = TfidfVectorizer(max_features=200, min_df=17, max_df=0.8 )
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Convert sparse matrix to array to use in classifiers
X_train_array = X_train_tfidf.toarray()
X_test_array = X_test_tfidf.toarray()

# Logistic Regresion Model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_array, y_train)

#Prediction
y_pred_train_lr = logistic_regression.predict(X_train_array)
y_pred_test_lr = logistic_regression.predict(X_test_array)

# Evaluate the model
accuracy_train_lr = accuracy_score(y_train, y_pred_train_lr)
accuracy_test_lr = accuracy_score(y_test, y_pred_test_lr)

#Print the accuracy
print('Logistic Regression - accuracy_train:', accuracy_train_lr)
print('Logistic Regression - accuracy_test:', accuracy_test_lr)

#Save models
folder_path = 'src/models/save_models'
joblib.dump(logistic_regression, os.path.join(folder_path, 'logistic_regression_model.pkl'))
joblib.dump(tfidf, os.path.join(folder_path, 'tfidf_vectorizer.pkl'))




