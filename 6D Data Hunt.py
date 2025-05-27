import matplotlib.pyplot as plt
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
from sklearn.decomposition import PCA
import numpy as np

rdd = sc.textFile("space.dat")

parsed_rdd = rdd.map(lambda line: [float(x.strip()) for x in line.split(',')])

# List to store the cost for each k
costs = []

# Run K-means for different numbers of clusters
for clusters in range(1, 15):
    model = KMeans.train(parsed_rdd, clusters, seed = 1)
    cost = model.computeCost(parsed_rdd)
    costs.append(cost)
    print(f"Clusters: {clusters}, Cost: {cost}")

# Create the elbow plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), costs, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Cost (Within-Cluster Sum of Squared Errors)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Save the plot
plt.savefig("kmeans_elbow_plot.png", dpi=300, bbox_inches='tight')
print("Elbow plot saved as 'kmeans_elbow_plot.png'")

# Train the K-means model with 6 clusters
k = 6
model = KMeans.train(parsed_rdd, k, seed =1)

# Predict clusters for each point
clustered_rdd = parsed_rdd.map(lambda point: (model.predict(point), point))

# Collect data for each cluster
cluster_data = clustered_rdd.groupByKey().mapValues(list).collect()

# Function to perform PCA and plot results
def pca_and_plot(cluster_points, cluster_id, dimensions=1):
    # Convert to numpy array
    points_array = np.array(cluster_points)
    
    # Perform PCA
    pca = PCA(n_components=dimensions)
    reduced_data = pca.fit_transform(points_array)
    
    # Plot
    fig = plt.figure(figsize=(10, 6))
    if dimensions == 1:
        plt.hist(reduced_data, bins=30, edgecolor='black')
        plt.xlabel('First Principal Component')
        plt.ylabel('Frequency')
    elif dimensions == 2:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
    
    plt.title(f'PCA Reduced Data for Cluster {cluster_id} ({dimensions}D)')
    plt.savefig(f'pca_cluster_{cluster_id}_{dimensions}D.png', dpi=300, bbox_inches='tight')
    plt.close()

# Perform PCA and plot for each cluster
for cluster_id, points in cluster_data:
    pca_and_plot(points, cluster_id, dimensions=1)  # For 1D
    pca_and_plot(points, cluster_id, dimensions=2)  # For 2D
    pca_and_plot(points, cluster_id, dimensions=3)  # For 3D

print("PCA plots for each cluster have been saved.")

def analyze_cluster_dimensionality(cluster_points):
    pca = PCA()
    pca.fit(cluster_points)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Determine effective dimensionality (98% of variance explained)
    effective_dim = np.argmax(cumulative_variance_ratio >= 0.98) + 1
    
    return effective_dim, explained_variance_ratio, pca

def analyze_cluster_stats(cluster_points, pca):
    mean = np.mean(cluster_points, axis=0)
    std_dev = np.std(cluster_points, axis=0)
    min_vals = np.min(cluster_points, axis=0)
    max_vals = np.max(cluster_points, axis=0)
    
    return mean, std_dev, min_vals, max_vals

# Analyze each cluster
for cluster_id, points in cluster_data:
    points_array = np.array(points)
    effective_dim, explained_variance_ratio, pca = analyze_cluster_dimensionality(points_array)
    mean, std_dev, min_vals, max_vals = analyze_cluster_stats(points_array, pca)
    num_points = len(points)
    
    print(f"Cluster {cluster_id}:")
    print(f"Number of points: {num_points}")
    print(f"Effective dimensionality: {effective_dim}")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f}")
    print(f"Mean: {mean}")
    print(f"Min values: {min_vals}")
    print(f"Max values: {max_vals}")
    print("\n")