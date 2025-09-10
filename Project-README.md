# 📰 Fake News Detection – Data Science Project

This project detects whether a given news article is **Fake** or **Real** using **Machine Learning & NLP**.  
It follows a complete **Data Science pipeline** – from dataset download, EDA, preprocessing, model building, evaluation, and a user-friendly **GUI** for real-time prediction.

---

## 📌 Project Overview

- **Problem Type:** Binary Text Classification (Fake = 0, Real = 1)
- **Domain:** Natural Language Processing (NLP)
- **Goal:** Build a model that can automatically classify news articles as **Fake** or **Real**

---

## 📂 Dataset

We use the [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

- **Fake.csv** → Fake news articles  
- **True.csv** → Real news articles  

Each file contains:
- `title` → Short news headline  
- `text` → Full article content  
- `subject` → News category  
- `date` → Publication date  

---

## 📊 Exploratory Data Analysis (EDA)

Steps performed:
- ✅ Dataset shape & basic statistics  
- ✅ Class distribution (Fake vs Real)  
- ✅ Text length distribution  
- ✅ WordClouds for Fake & Real news  
- ✅ Key insights from dataset (Fake news often uses more emotional/shocking words)

---

## 🧹 Data Preprocessing

- Convert text to lowercase  
- Remove punctuation, numbers, special symbols  
- Remove stopwords (like *the, is, and*)  
- Apply tokenization & lemmatization  
- Convert to numerical format using **TF-IDF Vectorization**  

---

## 🤖 Model Building & Training

Machine Learning models tested:
- **Logistic Regression** ✅ (Best Accuracy)
- Naive Bayes
- Random Forest (optional)

Training process:
- 80% Training, 20% Testing split  
- TF-IDF Vectorization (max 5000 features)
- Logistic Regression trained on vectorized data

---

## 📏 Model Evaluation

Metrics used:
- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix (visualized as heatmap)**

Sample output:
Accuracy: 0.98
Precision, Recall, F1-score → displayed per class (Fake/Real)


---

## 🖼 GUI for Prediction

We use **Tkinter** to build a simple but beautiful GUI (Red/Blue/White theme).

- 📝 Enter/paste a news article  
- 🎯 Click **Check News** button  
- ✅ Model predicts **REAL NEWS** (blue) or **FAKE NEWS** (red)  
- 🗂 Results are saved to `results_log.csv` for analysis  

---

## 📦 Project Structure



📁 fake-news-detection/
├── data/ # Fake.csv, True.csv (downloaded via kagglehub)
├── fake_news_detection.py # Full Data Science + GUI code
├── results_log.csv # Prediction log (auto-generated)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🔧 Installation & Setup

1. **Clone repository / open folder**
2. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud kagglehub


Run the script:

python fake_news_detection.py


GUI window will open for interactive predictions.

📈 Results & Insights

✅ Achieved ~98% accuracy with Logistic Regression + TF-IDF

✅ Discovered Fake news uses more emotionally charged words

✅ Built an interactive GUI for real-time detection

✅ Logged predictions for later analysis

🚀 Future Improvements

Use Word Embeddings (Word2Vec, GloVe, BERT)

Deploy as a Flask/Django Web App

Experiment with Deep Learning models (LSTM, Transformer, BERT)

📌 Dataset Link

🔗 Fake and Real News Dataset – Kaggle

🧑‍💻 Author

👤 Muhammad Sameer
