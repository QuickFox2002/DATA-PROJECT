
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset (direct link)
url = "https://calmcode.io/static/data/titanic.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# Show general info and missing values
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Data Cleaning: remove duplicates
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"\nRemoved {before-after} duplicate rows")

# Handle missing values: fill with median or mode where needed
if 'age' in df.columns:
    df['age'].fillna(df['age'].median(), inplace=True)

if 'embarked' in df.columns:
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

if 'fare' in df.columns:
    df['fare'].fillna(df['fare'].median(), inplace=True)

# Convert categorical variables into numeric
if 'sex' in df.columns:
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})

if 'embarked' in df.columns:
    df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Set style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Plot histogram of Age distribution
if 'age' in df.columns:
    plt.figure()
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title("Age Distribution of Passengers")
    plt.show()

# Plot count of Survival by Gender
if 'survived' in df.columns and 'sex' in df.columns:
    plt.figure()
    sns.countplot(x='survived', hue='sex', data=df)
    plt.title("Survival by Gender (0 = Male, 1 = Female)")
    plt.show()

# Plot boxplot of Fare vs Passenger Class
if 'pclass' in df.columns and 'fare' in df.columns:
    plt.figure()
    sns.boxplot(x='pclass', y='fare', data=df)
    plt.title("Fare Distribution by Passenger Class")
    plt.show()

# Heatmap of numeric correlations
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Statistical insights: summary values
print("\n--- Statistical Insights ---")
if 'age' in df.columns:
    print("Mean Age:", df['age'].mean())
    print("Median Age:", df['age'].median())
    print("Mode Age:", df['age'].mode()[0])
    print("Variance of Age:", df['age'].var())
    print("Standard Deviation of Age:", df['age'].std())

if 'fare' in df.columns:
    print("\nMean Fare:", df['fare'].mean())
    print("Median Fare:", df['fare'].median())
    print("Mode Fare:", df['fare'].mode()[0])

if 'survived' in df.columns:
    print("\nCorrelation with Survival:\n", numeric_df.corr()['survived'].sort_values(ascending=False))

# Print outcome summary
print("\n--- Key Insights ---")
if 'survived' in df.columns:
    print(f"Survival Rate: {df['survived'].mean():.2%}")
    print("Women and higher-class passengers had higher chances of survival.")
