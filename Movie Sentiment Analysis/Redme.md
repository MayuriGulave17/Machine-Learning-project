
## 🎬 Movie Review Sentiment Analysis (Pure NLP Project)



## Overview
This project performs **sentiment analysis** on movie reviews using **Natural Language Processing (NLP)** techniques.  
The goal is to classify reviews as **Positive** or **Negative** by processing and analyzing the text directly using NLP-based methods.  
## Objective
- Clean and preprocess raw text data.  
- Apply NLP techniques like tokenization, stopword removal, and lemmatization.  
- Extract features using **Bag of Words (BoW)** and **TF-IDF**.  
- Train a baseline sentiment classifier (Logistic Regression / Naive Bayes) only with NLP-driven features.  
- Evaluate performance using classification metrics.  
## 📂 Dataset  
- **IMDB Movie Reviews Dataset**  
- Contains:  
  - Review Text (raw text)  
  - Sentiment Label (Positive / Negative)  
## ## 🛠️ Methodology  
1. **Text Preprocessing**  
   - Remove punctuation, numbers, and special characters.  
   - Convert text to lowercase.  
   - Tokenize sentences into words.  
   - Remove stopwords (common non-informative words like *the, is, and*).  
   - Apply **stemming / lemmatization** to reduce words to their root form.  

2. **Feature Extraction**  
   - Bag of Words (BoW) representation  
   - TF-IDF (Term Frequency – Inverse Document Frequency)  

3. **Modeling (using NLP features)**  
   - Logistic Regression (baseline)  
   - Naive Bayes (common for text classification)  

4. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix  
## 📊 Results  
- Logistic Regression (TF-IDF): 89% Accuracy  
- Naive Bayes (BoW): 85% Accuracy  

📌 **Best Model** → Logistic Regression (TF-IDF).  

## ##  Tech Stack    
- Python   
 
- Scikit-learn (for TF-IDF, BoW, and classifiers)  
- Pandas, NumPy (data handling)  
