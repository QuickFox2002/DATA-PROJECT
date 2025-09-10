import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# DOWNLOAD DATASET

print("Downloading dataset...")
path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
print("✅ Dataset downloaded to:", path)

# Load Fake & True CSV
fake = pd.read_csv(f"{path}/Fake.csv")
true = pd.read_csv(f"{path}/True.csv")

# Add labels
fake["label"] = 0  # Fake = 0
true["label"] = 1  # Real = 1

# Combine datasets
df = pd.concat([fake, true], ignore_index=True)
print("✅ Combined Dataset Shape:", df.shape)


# EDA

print("\nPerforming EDA...")

# Class Distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df, palette="coolwarm")
plt.title("Fake vs Real News Distribution")
plt.xlabel("Label (0=Fake, 1=Real)")
plt.ylabel("Count")
plt.show()

#  Text Length Distribution
df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="text_length", hue="label", kde=True, palette="coolwarm", bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# 3. Word Clouds
fake_text = " ".join(fake["text"].astype(str).tolist())
true_text = " ".join(true["text"].astype(str).tolist())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
wc_fake = WordCloud(width=500, height=300, background_color="white", colormap="Reds").generate(fake_text)
plt.imshow(wc_fake, interpolation="bilinear")
plt.axis("off")
plt.title("Fake News WordCloud")

plt.subplot(1, 2, 2)
wc_true = WordCloud(width=500, height=300, background_color="white", colormap="Blues").generate(true_text)
plt.imshow(wc_true, interpolation="bilinear")
plt.axis("off")
plt.title("Real News WordCloud")
plt.show()


#  DATA PREPROCESSING

print("\nPreprocessing text...")

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)


# TRAIN-TEST SPLIT & TF-IDF
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


# MODEL TRAINING

print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# MODEL EVALUATION
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix")
plt.show()

# GUI FOR PREDICTION
def predict_news():
    input_text = text_area.get("1.0", tk.END).strip()
    if input_text == "":
        messagebox.showwarning("Warning", "Please enter news text!")
        return
    input_clean = clean_text(input_text)
    input_vector = vectorizer.transform([input_clean])
    prediction = model.predict(input_vector)[0]
    result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"
    result_label.config(text=f"Prediction: {result}", fg="blue" if prediction == 1 else "red")
    # Log result
    with open("results_log.csv", "a", encoding="utf-8") as f:
        f.write(f"\"{input_text}\",{prediction}\n")

# GUI Window
root = tk.Tk()
root.title("📰 Fake News Detector (Data Science Edition)")
root.geometry("650x400")
root.config(bg="white")

title_label = tk.Label(root, text="Fake News Detection (ML Model)", font=("Arial", 16, "bold"), bg="white", fg="blue")
title_label.pack(pady=10)

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10, font=("Arial", 11))
text_area.pack(pady=10)

predict_button = tk.Button(root, text="Check News", command=predict_news, bg="red", fg="white", font=("Arial", 12, "bold"))
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="white")
result_label.pack(pady=10)

root.mainloop()
