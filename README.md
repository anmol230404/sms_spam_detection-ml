SMS Spam Detection using Machine Learning

Project Overview

This project classifies SMS messages as SPAM or HAM (genuine messages) using Machine Learning algorithms.

Objective

To develop a text classification model that identifies unwanted or spam SMS messages.

Dataset

- Source: "SMS Spam Collection Dataset" (https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv)
- Original messages: 5,572
- Messages after duplicate removal: 5,169
- HAM messages: 4,825 before duplicate removal
- SPAM messages: 747 before duplicate removal

Technologies and Libraries

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- TF-IDF Vectorization

Machine Learning Algorithms

1. Multinomial Naive Bayes
2. Logistic Regression

Methodology

1. Load the dataset.
2. Check missing values and remove duplicate rows.
3. Encode labels: HAM = 0 and SPAM = 1.
4. Split the data into 80% training and 20% testing.
5. Convert text into numerical features using TF-IDF.
6. Train and evaluate both classifiers.
7. Compare accuracy, precision, recall, F1-score, and confusion matrices.
8. Save the selected model and TF-IDF vectorizer.

Results

Model| Accuracy
Multinomial Naive Bayes| 96.71%
Logistic Regression| 95.65%

Naive Bayes performed better on this test split. Its SPAM recall was approximately 74%, so some spam messages were missed.

Sample Predictions

- Promotional prize message: SPAM
- Normal lunch-planning message: HAM

Saved Model Files

- "spam_detection_model/naive_bayes_model.pkl"
- "spam_detection_model/tfidf_vectorizer.pkl"

How to Run

Open "Spam_Ham_Classifier.ipynb" in Google Colab and execute the cells in order.

Limitations

The model may misclassify some messages. Its results depend on the dataset and test split, and predictions are not guaranteed to be correct.
