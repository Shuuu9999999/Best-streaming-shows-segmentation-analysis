from sklearn.decomposition import PCA
import numpy as np
# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
# Create a DataFrame for the reduced data
reduced_df = pd.DataFrame(reduced_matrix, columns=['PC1', 'PC2'])
reduced_df['cluster'] = clusters

# Plot the 2D clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=reduced_df, palette='viridis', s=100)
plt.title('2D Plot of Clusters (PCA)')
plt.show()
# Import necessary libraries
from sklearn.cluster import KMeans
import pandas as pd
# Assuming tfidf_matrix and movies_df are already defined
# Create a KMeans object with 5 clusters and fit the model
km = KMeans(n_clusters=5)
km.fit(tfidf_matrix)
# Get the cluster labels for each movie
clusters = km.labels_.tolist()
# Add the cluster information to the DataFrame
movies_df["cluster"] = clusters
# Display the number of films per cluster
cluster_counts = movies_df['cluster'].value_counts()
print(cluster_counts)
# Plot the counts for better visualization
import matplotlib.pyplot as plt
cluster_counts.plot(kind='bar')
plt.title('Number of Films per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Films')
plt.show()
# Function to print movies in each cluster
def print_movies_in_clusters(df, cluster_label_column='cluster', title_column='title'):
    for cluster in sorted(df[cluster_label_column].unique()):
        print(f"\nCluster {cluster}:")
        movies_in_cluster = df[df[cluster_label_column] == cluster][title_column].tolist()
        for movie in movies_in_cluster:
            print(movie)
# Print the names of the movies in each cluster
print_movies_in_clusters(movies_df)
# Function to find similar movies in the same cluster
def find_similar_movies(movie_title, df, cluster_label_column='cluster', title_column='title'):
    # Check if the movie exists in the DataFrame
    if movie_title not in df[title_column].values:
        return f"Movie '{movie_title}' not found in the dataset."
      # Get the cluster label of the given movie
    cluster_label = df[df[title_column] == movie_title][cluster_label_column].values[0]
      # Get all movies in the same cluster
    similar_movies = df[df[cluster_label_column] == cluster_label][title_column].tolist()
     # Remove the given movie from the list
    if movie_title in similar_movies:
        similar_movies.remove(movie_title)
     return similar_movies
# Example usage: Prompt user for a movie title and print similar movies
movie_title = input("Enter a movie title: ")  # User input for movie title
similar_movies = find_similar_movies(movie_title, movies_df)
if isinstance(similar_movies, list):
    print(f"\nMovies similar to '{movie_title}':")
    for movie in similar_movies:
        print(movie)
else:
    print(similar_movies)
