import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

new_point, _ = make_blobs(n_samples=1, centers=1, random_state=1)
cluster = kmeans.predict(new_point)[0]

print("Cluster centroids:\n", centroids)
print("New data point:", new_point)
print("Belongs to cluster:", cluster)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.scatter(new_point[0][0], new_point[0][1], c='black', s=200, marker='*', label='New Point')
plt.legend()
plt.title("KMeans Clustering")
plt.show()