#load libraries
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Load the test data
df = pd.read_csv("data/labeling_data.csv")
df.dropna(subset=['final', 'sentiment'], inplace=True)

# Split the data into features and target variable
X = df['final']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Load the TfidfVectorizer from the trained model
tfidf = joblib.load('src/models/save_models/tfidf_vectorizer.pkl')

# Transform the test data
X_test_tfidf = tfidf.transform(X_test)

# Load the trained models (Naive Bayes, Random Forest, and Logistic Regression)
naive_bayes = joblib.load('src/models/save_models/naive_bayes_model.pkl')
random_forest = joblib.load('src/models/save_models/random_forest_model.pkl')
logistic_regression = joblib.load('src/models/save_models/logistic_regression_model.pkl')

# Make predictions with each model
y_pred_nb = naive_bayes.predict(X_test_tfidf)
y_pred_rf = random_forest.predict(X_test_tfidf)
y_pred_lr = logistic_regression.predict(X_test_tfidf)

# Evaluate the models using accuracy, precision, recall, and F1-score
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Print evaluation for each model
print("Naive Bayes Model Evaluation:")
evaluate_model(y_test, y_pred_nb)

print("Random Forest Model Evaluation:")
evaluate_model(y_test, y_pred_rf)

print("Logistic Regression Model Evaluation:")
evaluate_model(y_test, y_pred_lr)
