
# Medical Texts Classification

An NLP pipeline for classifying Arabic medical question-answer texts into 8 medical specialties, covering the full workflow from Arabic-specific text preprocessing to classical ML and deep learning models.

---

## Overview

Arabic medical Q&A platforms generate large volumes of unstructured text that need to be routed to the right medical specialty. This project builds and compares multiple text classification approaches — from traditional Bag-of-Words models to deep neural networks — specifically tuned for the linguistic challenges of Arabic medical text.

**Target classes (8 medical specialties):** High Blood Pressure, Psychiatry, Benign/Malignant Tumors, Diabetes, Wounds, Bone Injuries, Blood Disorders, Endocrine Disorders.

---

## Key Features

### Arabic-Specific Text Preprocessing
Built a modular pipeline of 17 custom preprocessing functions, including:
- Doctor name/specialization removal, link and date-time stripping
- Indic-to-Arabic digit normalization
- Diacritics removal, Hamza/Alif standardization, Tatweel (kashida) removal
- Duplicate character/space normalization
- Arabic stopword removal and ISRI stemming
- Systematic evaluation of each preprocessing step's individual impact on model accuracy

### Exploratory Text Analysis
- Word frequency, hapax legomena, and n-gram (bigram/trigram) collocation analysis per class
- Arabic word clouds (using `arabic_reshaper` and `python-bidi` for correct RTL rendering)
- Vocabulary size and class distribution analysis

### Feature Engineering
- Bag-of-Words (CountVectorizer) and TF-IDF representations
- Custom Skip-gram word embeddings trained from scratch (Keras), including word analogy testing (e.g., "high" − "pressure" + "blood")
- Hyperparameter tuning via GridSearchCV (vectorizer and Logistic Regression parameters)

### Models Implemented & Compared
| Approach | Technique |
|---|---|
| Baseline | Logistic Regression (multinomial, BoW) |
| Optimized preprocessing | Logistic Regression on best preprocessing pipeline |
| TF-IDF | Logistic Regression with TF-IDF + GridSearchCV |
| Deep Learning | DNN on Bag-of-Words |
| Deep Learning | DNN on TF-IDF |
| Deep Learning | CNN with learned embeddings (Conv1D + MaxPooling) |

All models were evaluated using **accuracy** and **balanced accuracy** (to account for class imbalance across specialties), with results consolidated into a comparison table.

---

## Tech Stack

**NLP:** NLTK (tokenization, stemming, stopwords, collocations), arabic_reshaper, python-bidi, WordCloud

**Feature Extraction:** Scikit-learn (CountVectorizer, TfidfVectorizer), Keras Tokenizer

**Machine Learning:** Scikit-learn (Logistic Regression, GridSearchCV)

**Deep Learning:** Keras/TensorFlow (Dense networks, Conv1D, Embedding layers, Skip-gram)

**Data Handling:** Pandas, NumPy

**Visualization:** Matplotlib, Seaborn

---

## Project Structure

```
Medical_Texts_Classification.ipynb   # Full pipeline: preprocessing → feature engineering → modeling → evaluation
homework1_data/                       # train/val/test datasets
```

---

## Key Findings

- Systematic ablation of preprocessing steps showed that not all cleaning operations improve accuracy equally — some (e.g., aggressive stemming) can reduce performance depending on the model.
- TF-IDF with tuned Logistic Regression outperformed raw Bag-of-Words baselines.
- Deep learning models (DNN, CNN) were evaluated against classical baselines to assess the trade-off between model complexity and performance on a relatively small, imbalanced Arabic medical dataset.

---.
