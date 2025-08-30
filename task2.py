import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset from a local CSV file
df = pd.read_csv('Wholesale customers data.csv.csv')

# Data Cleaning: Handle missing values
df.fillna(df.mean(), inplace=True)

# Select features for clustering
X = df[['Grocery', 'Frozen', 'Milk', 'Detergents_Paper', 'Delicassen']]

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Grocery', y='Frozen', hue='Cluster', palette='Set1')
plt.title('Customer Segmentation by Grocery and Frozen Purchases')
plt.xlabel('Grocery Spend')
plt.ylabel('Frozen Spend')
plt.show()

# Interpret Clusters
for i in range(5):
    cluster = df[df['Cluster'] == i]
    print(f"Cluster {i}:")
    print(cluster[['Grocery', 'Frozen', 'Milk', 'Detergents_Paper', 'Delicassen']].describe())
    print("\n")
