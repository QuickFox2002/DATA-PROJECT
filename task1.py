import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Supermarket Sales dataset from CSV file
try:
    df = pd.read_csv('supermarket_sales.csv.csv')  # Adjust path if file is in another directory
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Display first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Step 3: Check for missing values before filling
print("\nMissing values before filling:")
print(df.isnull().sum())  # Display number of missing values in each column

# Step 4: Handle missing data (fill numeric columns with mean)
# Select only numeric columns to fill missing values with the mean
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 5: Check if missing values are handled after filling
print("\nMissing values after filling:")
print(df.isnull().sum())  # Check again for missing values

# Step 6: Exploratory Data Analysis (EDA)

# 1. Total Sales by Product Line
category_sales = df.groupby('Product line')['Total'].sum().sort_values(ascending=False)

# 2. Number of Customers by City
city_customers = df['City'].value_counts()

# 3. Average Purchase Value
avg_purchase = df['Total'].mean()

# Step 7: Visualizations

# 1. Total Sales by Product Line
plt.figure(figsize=(8,6))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()

# 2. Number of Customers by City
plt.figure(figsize=(8,6))
sns.countplot(x='City', data=df)
plt.title('Number of Customers by City')
plt.xlabel('City')
plt.ylabel('Count')
plt.show()

# 3. Distribution of Purchase Value
plt.figure(figsize=(6,6))
sns.histplot(df['Total'], bins=20, kde=True)
plt.title('Distribution of Purchase Value')
plt.xlabel('Total Purchase')
plt.ylabel('Frequency')
plt.show()

# Step 8: Insights
print(f"\nHighest sales category: {category_sales.idxmax()}")
print(f"City with most customers: {city_customers.idxmax()}")
print(f"Average purchase value: ${avg_purchase:.2f}")
