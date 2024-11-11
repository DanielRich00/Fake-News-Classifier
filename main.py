import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training, validation, and test data

train_data = pd.read_csv('train.csv', on_bad_lines='skip', delimiter=';')
validation_data = pd.read_csv('validation.csv', on_bad_lines = 'skip', delimiter=';')
test_data = pd.read_csv('test.csv', on_bad_lines = 'skip', delimiter=';')

# Combine title and text into a single column

train_data['content'] = train_data['title'] + " " + train_data['text']
validation_data['content'] = validation_data['title'] + " " + validation_data['text']
test_data['content'] = test_data['title'] + " " + test_data['text']


# seperating features and labels

X_train = train_data['content']
y_train = train_data['label']
X_val = validation_data['content']
y_val = validation_data['label']
X_test = test_data['content']
y_test = test_data['label']

# Initialize TF-IDF Vectorizer

tfidf = TfidfVectorizer(max_features=5000)

# Fit TF-IDF on training data and transform training, validation, and test data

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)

# Train the model on the training data

log_reg.fit(X_train_tfidf, y_train)

y_val_pred = log_reg.predict(X_val_tfidf)

# Evaluate performance

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# Predict on test set
y_test_pred = log_reg.predict(X_test_tfidf)

# Evaluate performance
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))