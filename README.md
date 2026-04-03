# Multi-Class Text Classification (NLP)

This project explores multi-class text classification using various word representation techniques combined with both Machine Learning (ML) and Neural Network (NN) models.

##  Overview
- Compared **4 text representation methods**:
  - Bag of Words (BoW)
  - TF-IDF
  - GloVe (pre-trained embeddings)
  - Word2Vec (Skip-gram)

- Evaluated **22 model combinations**:
  - ML: Logistic Regression, Naive Bayes, Random Forest
  - NN: DNN, RNN, GRU, LSTM, Bi-directional variants

##  Pipeline
1. **EDA**
   - Class distribution, text length, n-grams
2. **Preprocessing**
   - Lowercasing, tokenization, stopword removal
3. **Feature Engineering**
   - BoW, TF-IDF, GloVe, Skip-gram
4. **Model Training**
   - ML + Deep Learning architectures
5. **Evaluation**
   - Accuracy, F1-Macro, F1-Weighted

##  Key Results
-  Best Model: **GloVe + LSTM**
  - Accuracy: **71.38%**
  - F1 Score: **0.7081**
-  Worst Model: TF-IDF + Random Forest (~48.9%)

 Sequential models + contextual embeddings significantly outperformed frequency-based approaches. :contentReference[oaicite:0]{index=0}

## Insights
- Context-aware embeddings (GloVe) improve performance
- LSTM captures long-term dependencies effectively
- Logistic Regression is the strongest ML baseline
- SimpleRNN suffers from vanishing gradient issues

##  Tech Stack
- Python
- Scikit-learn
- TensorFlow / Keras
- NLP preprocessing (NLTK)

## Future Work
- Transformer-based models (BERT, etc.)
- Handling class imbalance
- Domain-specific embeddings

## 📎 Project Report
See full documentation: [Report.pdf](./Report.pdf)
