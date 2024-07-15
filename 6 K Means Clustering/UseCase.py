import numpy as np
from sklearn.datasets import make_blobs
from KMeansClass import KMeans

# Generate a synthetic dataset with centers, samples, and features
X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=42)
print("X shape: ", X.shape)

# Determine the number of unique clusters in the dataset
clusters = len(np.unique(y))
print("Number of K (clusters): ", clusters)  # Print the number of clusters

# Initialize the KMeans algorithm with the determined number of clusters and iterations
k = KMeans(K=clusters, maxIterations=150, showSteps=False)

y_pred = k.predict(X)

# Plot the final clusters and centroids
k.plot()
