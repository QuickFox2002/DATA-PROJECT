# Netflix Movies & TV Shows — EDA & ML Classification

This project performs **Exploratory Data Analysis (EDA)** and builds a **Machine Learning model** to classify whether a Netflix title is a **Movie** or a **TV Show** using metadata from the [Kaggle Netflix Movies & TV Shows Dataset](https://www.kaggle.com/shivamb/netflix-shows).

---

## 📂 Project Structure


---

## 📊 Dataset

**Source:** [Netflix Movies and TV Shows – Kaggle](https://www.kaggle.com/shivamb/netflix-shows)

**File:** `netflix_titles.csv`

| Column         | Description                                      |
|---------------|------------------------------------------------|
| `show_id`     | Unique ID for each title                       |
| `type`        | Movie or TV Show                               |
| `title`       | Name of the title                              |
| `director`    | Director name (if available)                   |
| `cast`        | Main cast (comma-separated)                    |
| `country`     | Country of production                          |
| `date_added`  | Date it was added to Netflix                   |
| `release_year`| Year of release                                |
| `rating`      | Age rating (e.g., PG, TV-MA)                   |
| `duration`    | Length in minutes (movies) or seasons (TV shows) |
| `listed_in`   | Genres / categories                            |
| `description` | Summary of the title                           |

---

## 🔍 Exploratory Data Analysis (EDA)

- **Missing Values:** Checked and filled with `"Unknown"` where appropriate.
- **Distribution of Type:** Bar plot (Movies vs TV Shows).
- **Release Year Distribution:** Histogram to visualize trends over time.
- **Top Production Countries:** Bar plot of top 15 countries.

📁 **Plots saved in `./plots/`**

---

## 🛠️ Feature Engineering

- `duration_min` → Converted to numeric minutes (estimated for TV Shows: `seasons * 10 episodes * 45 min`).
- `country_count` → Number of countries listed.
- `has_director`, `has_cast` → Binary flags.
- `year_bucket` → Release year grouped into bins (e.g., `2000-09`, `2015-19`).
- `title_len`, `desc_len` → Length of title/description.
- `main_genre` → Extracted first listed genre.
- One-hot encoding applied to `main_genre` and `year_bucket`.

---

## 🤖 Machine Learning Models

| Model               | Accuracy (Sample) |
|--------------------|-----------------|
| Logistic Regression| ~0.82 |
| Random Forest      | ~0.86 ✅ (Best Model) |

**Target Variable:**  
`0 = Movie`, `1 = TV Show`

