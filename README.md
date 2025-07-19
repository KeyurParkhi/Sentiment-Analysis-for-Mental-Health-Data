## Sentiment Analysis for Mental Health Data
Sentiment classification of mental health text data using Natural Language Processing and Machine Learning models

---

## Problem Statement
Mental health-related discussions are growing across online platforms, and there's an urgent need to automatically detect emotional sentiment in such conversations to facilitate timely interventions. This project focuses on building a machine learning pipeline that classifies user-generated mental health text into sentiment categories, helping in understanding the emotional state and potentially assisting healthcare support systems. The goal is to support **early understanding** of user sentiments through supervised machine learning and natural language processing for which we classify each text into:
- **Negative**
- **Neutral**
- **Positive**
--- 

## Tools and Technologies
- Languages & Libraries: Python, Pandas, NumPy, Matplotlib
- NLP Techniques: Tokenization, Lemmatization, TF-IDF Vectorization
- Models: Logistic Regression, Naïve Bayes, SVM
- Notebook: Google Colab / Jupyter Notebook
---

## Dataset
- Dataset Source: [Kaggle - NLP Mental Health Conversations](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations)
- The dataset contains **3512** text records of the conversations between the patient and the psychologist.
- Features: 'Context' and 'Response'
- Sentiment categories: **Negative**, **Neutral**, and **Positive**.
- Preprocessing steps:
  - Lowercasing
  - Removing punctuation/special characters
  - Tokenization
  - TF-IDF vectorization

---

## Project Workflow
1. Text Data Preprocessing:
- Cleaned raw text
- Removed stopwords
- Applied lemmatization
- Vectorized using TF-IDF

2. Model Training:
- Split data into train-test sets
- Trained classifiers (Logistic Regression, SVM, Naive-Bayes)

3. Model Evaluation:
- Compared model performance using classification reports
- Analyzed overfitting via train-test accuracy gap
- Visualized confusion matrices

4. Best Performing Model:
- Identified top-performing model using appropriate evaluation metrics

--- 
## Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Naïve Bayes
Each model was trained using the TF-IDF matrix and evaluated using accuracy, precision, recall, F1-score and confusion matrix.
---

## Model Evaluation

### 1. Logistic Regression
- **Accuracy**: `95.29%`
- **Classification Report**
| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| **Negative**  | 0.97      | 0.88   | 0.92     | 129     |
| **Neutral**   | 0.95      | 1.00   | 0.97     | 486     |
| **Positive**  | 0.99      | 0.82   | 0.89     | 87      |
|               |           |        |          |         |
| **Accuracy**  |           |        | **0.95** | 702     |
| **Macro avg** | 0.97      | 0.90   | 0.93     | 702     |
| **Weighted avg** | 0.95   | 0.95   | 0.95     | 702     |

<pre> Confusion Matrix:
[[114  14   1]
 [  2 484   0]
 [  2  14  71]] </pre>


 ### 2. Support Vector Machines
- **Accuracy**: `97.86%`
- **Classification Report**
| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| **Negative**    | 0.97      | 0.95   | 0.96     | 129     |
| **Neutral**     | 0.98      | 0.99   | 0.99     | 486     |
| **Positive**    | 0.98      | 0.95   | 0.97     | 87      |
|                 |           |        |          |         |
| **Accuracy**    |           |        | **0.98** | 702     |
| **Macro avg**   | 0.98      | 0.96   | 0.97     | 702     |
| **Weighted avg**| 0.98      | 0.98   | 0.98     | 702     |

<pre> Confusion Matrix**:
[[122   7   0]
 [  2 482   2]
 [  2   2  83]] </pre>

 ### 3. Naive-Bayes
- **Accuracy**: `79.20%`
- **Classification Report**
| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| **Negative**    | 1.00      | 0.38   | 0.55     | 129     |
| **Neutral**     | 0.77      | 1.00   | 0.87     | 486     |
| **Positive**    | 1.00      | 0.24   | 0.39     | 87      |
|                 |           |        |          |         |
| **Accuracy**    |           |        | **0.79** | 702     |
| **Macro avg**   | 0.92      | 0.54   | 0.60     | 702     |
| **Weighted avg**| 0.84      | 0.79   | 0.75     | 702     |

<pre> Confusion Matrix**:
[[ 49  80   0]
 [  0 486   0]
 [  0  66  21]] </pre>

---
## Conclusion
- **SVM** achieved the **best performance** with the highest weighted F1-Score.
- **Logistic Regression** performed comparatively well.
- **Naïve Bayes** struggled to capture positive sentiments and had the weakest recall.
This study shows that **machine learning** can be effectively applied to **mental health sentiment analysis**, and models like SVM can assist in better understanding mental health scenarios.

--- 
## Team Members:
- Keyur Parkhi
- Gourish Salgaonkar
- Dev Vatnani
