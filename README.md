# Fake News Classification with Logistic Regression
This project implements a machine learning model to classify news articles as real or fake based on their content. It uses the Logistic Regression model along with TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction. The project leverages Pandas, scikit-learn, and other libraries to preprocess the data, train the model, and evaluate its performance.

## Project Overview
The goal of this project is to classify news articles into two categories: real or fake. We use the title and text of news articles as input features and apply a logistic regression model to predict the label.

## File Structure
train.csv: The training dataset containing news articles and their labels (real/fake).
validation.csv: The validation dataset used to evaluate the model during training.
test.csv: The test dataset used to evaluate the final performance of the trained model.
main.py: The main Python script that implements the entire process: loading data, preprocessing, training the model, and evaluating performance.

## Libraries Used
pandas: For data manipulation and analysis.
scikit-learn: For machine learning tools, including model training, evaluation, and text vectorization.
matplotlib: For plotting graphs and visualizing performance metrics.
Setup

## How to Use
Prepare your data: Make sure you have the following CSV files:

1. train.csv: The training data with news articles and labels.
2. validation.csv: The validation data used for tuning the model.
3. test.csv: The test data used for final evaluation.
Run the model: Run the main.py script to train and evaluate the logistic regression model on your datasets

## Results: The script will print the following performance metrics on the validation and test datasets:
Accuracy
Precision, Recall, F1-score
Confusion Matrix
