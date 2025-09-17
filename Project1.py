
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# data & ML
import kagglehub
import pandas as pd
import numpy as np
from datetime import datetime

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

sns.set(style="whitegrid")


# 1) Download dataset (kagglehub)
print("Downloading Netflix dataset from Kaggle (shivamb/netflix-shows)...")
try:
    data_dir = kagglehub.dataset_download("shivamb/netflix-shows")
except Exception as e:
    print("Error during kagglehub.dataset_download():", e)
    print("Make sure kagglehub is installed and you ran `kagglehub login` once.")
    raise

csv_path = os.path.join(data_dir, "netflix_titles.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find netflix_titles.csv at {csv_path}")

print("Loaded dataset at:", csv_path)
df = pd.read_csv(csv_path)
print("Raw shape:", df.shape)
print(df.columns.tolist())
print(df.head())


# 2) Quick cleaning & overview
print("\n--- missing values ---")
print(df.isnull().sum())

# Keep a copy for EDA plots
df_eda = df.copy()

# Convert date_added to datetime (some rows NaN)
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Fill missing country/cast/director with 'Unknown' (safe)
for c in ['country','cast','director','rating','duration','listed_in']:
    if c in df.columns:
        df[c] = df[c].fillna('Unknown')

# Standardize release_year as int
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').astype('Int64')


# 3) EDA (basic plots saved to ./plots)
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df)
plt.title('Count: Movies vs TV Shows')
plt.tight_layout(); plt.savefig("plots/type_count.png"); plt.close()

plt.figure(figsize=(10,5))
sns.histplot(df['release_year'].dropna().astype(int), bins=30)
plt.title('Distribution of Release Year')
plt.tight_layout(); plt.savefig("plots/release_year_hist.png"); plt.close()

# Top 15 countries
top_countries = (df[df['country'] != 'Unknown']
                 .country.str.split(',').explode().str.strip()
                 .value_counts().head(15))
plt.figure(figsize=(8,6))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 15 Production Countries')
plt.tight_layout(); plt.savefig("plots/top_countries.png"); plt.close()

print("Saved basic EDA plots to ./plots/")


# 4) Feature engineering
df_fe = df.copy()

# 4.1: duration: unify to numeric minutes (Movies: "90 min", TV Shows: "3 Seasons")
def parse_duration(row):
    dur = row.get('duration', '')
    if pd.isna(dur) or dur == 'Unknown':
        return np.nan
    dur = str(dur).strip()
    if 'min' in dur:
        try:
            return int(dur.replace('min','').strip())
        except:
            return np.nan
    # seasons -> convert to approximate minutes: seasons * 10 episodes * 45 min ~ rough
    if 'Season' in dur or 'Seasons' in dur:
        try:
            n = int(dur.split()[0])
            return n * 10 * 45
        except:
            return np.nan
    return np.nan

df_fe['duration_min'] = df_fe.apply(parse_duration, axis=1)

# 4.2: country_count (how many countries listed)
df_fe['country_count'] = df_fe['country'].apply(lambda x: 0 if x=='Unknown' or pd.isna(x) else len([c.strip() for c in str(x).split(',')]))

# 4.3: has_director, has_cast
df_fe['has_director'] = (~df_fe['director'].isin(['Unknown',''])) .astype(int)
df_fe['has_cast'] = (~df_fe['cast'].isin(['Unknown',''])) .astype(int)

# 4.4: year_bucket (group release_year)
df_fe['rel_year'] = df_fe['release_year'].fillna(0).astype(int)
df_fe['year_bucket'] = pd.cut(df_fe['rel_year'].replace(0, np.nan), bins=[0,1950,1980,2000,2010,2015,2020,2025],
                              labels=['<1950','1950-79','1980-99','2000-09','2010-14','2015-19','2020+'])

# 4.5: length of title & description
df_fe['title_len'] = df_fe['title'].astype(str).apply(len)
df_fe['desc_len'] = df_fe['description'].astype(str).apply(lambda x: len(x))

# 4.6: main_genre (take first listed genre)
df_fe['main_genre'] = df_fe['listed_in'].apply(lambda x: str(x).split(',')[0].strip() if x!='Unknown' else 'Unknown')

print("Engineered columns example:")
print(df_fe[['title','type','duration','duration_min','country','country_count','rel_year','year_bucket','main_genre']].head())


# 5) Prepare dataset for modeling
df_model = df_fe.copy()
df_model = df_model[~df_model['type'].isnull()]
df_model['target'] = (df_model['type'].str.strip().str.lower() == 'tv show').astype(int)

# Select features
features = [
    'duration_min', 'country_count', 'has_director', 'has_cast',
    'title_len', 'desc_len', 'rel_year'
]
df_model = pd.get_dummies(df_model, columns=['main_genre','year_bucket'], prefix=['genre','year'], dummy_na=False)
features += [c for c in df_model.columns if c.startswith('genre_') or c.startswith('year_')]

X = df_model[features].copy()
y = df_model['target'].copy()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X = X.fillna(0)

print("Final feature matrix shape:", X.shape)


# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)


# 7) Scale numeric features (within pipeline for LR)
numeric_features = num_cols
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])


# 8) Train models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    if name == "LogisticRegression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()

    results[name] = {'model': model, 'accuracy': acc, 'probs': probs}


# 9) Feature importance for RandomForest
rf = results['RandomForest']['model']
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6)); sns.barplot(x=importances.values, y=importances.index)
plt.title("Top features (RandomForest)"); plt.tight_layout(); plt.show()


# 10) Save best model
best_name = max(results.keys(), key=lambda n: results[n]['accuracy'])
best_model = results[best_name]['model']
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", f"netflix_type_{best_name}.pkl")
joblib.dump(best_model, model_path)
print(f"Saved best model ({best_name}) -> {model_path}")


# 11) QUICK NOTES
print("\nDone. Notes:")
print("- Target: 0 = Movie, 1 = TV Show")
print("- Features used (numeric + one-hot genres/year buckets).")
print("- For production: consider text features (cast/director/description) using NLP/vectorization.")
print("- Dataset source: Kaggle - 'Netflix Movies and TV Shows' (shivamb/netflix-shows).")
