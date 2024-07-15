import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans():
    def __init__(self, K=3, maxIterations=100, showSteps=False):
        self.K = K  # Number of clusters
        self.maxIterations = maxIterations  # Maximum number of iterations
        self.showSteps = showSteps  # Flag to plot the steps
        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # The centers (mean vector) for each cluster
        self.centroids = []

    # Initialize random centroids by choosing K random samples (One Time Use Only)
    def _initializeRandomCentroids(self, indices):
        chosenIndices = np.random.choice(indices, self.K, replace=False)
        return chosenIndices

    # Main method to perform KMeans clustering
    def predict(self, X):
        self.X = X  # Dataset
        numOfSamples, numOfFeatures = X.shape  # Number of samples and features

        # Initialize centroids by choosing K random samples
        randomIndices = self._initializeRandomCentroids(numOfSamples)
        self.centroids = [X[idx] for idx in randomIndices]
        flag = True  # Flag for plotting

        # Iterate to optimize clusters
        for _ in range(self.maxIterations):
            # Assign samples to the closest centroids
            self.clusters = self._assignCentroids(self.centroids)

            # Plot the clusters if showSteps is True
            if self.showSteps and flag == True:
                flag = False
                self.plot()

            # Store the previous centroids
            prevCentroids = self.centroids
            # Calculate new centroids from the clusters
            self.centroids = self._updateCentroids(self.clusters, numOfFeatures)

            # Check for convergence if prevCentroids and newCentroids are same or not
            if self._algoConvergence(prevCentroids, self.centroids):
                break

            # Plot the clusters if showSteps is True
            if self.showSteps:
                self.plot()

        # Return cluster labels for each sample
        return self._assignClusterLabels(self.clusters, numOfSamples)

    # Assign cluster labels to each sample
    def _assignClusterLabels(self, clusters, numOfSamples):
        labels = np.empty(numOfSamples)
        for clusterIdx, clusterPointsIndices in enumerate(clusters):
            for sampleIdx in clusterPointsIndices:
                labels[sampleIdx] = clusterIdx
        return labels

    # Assign samples to the closest centroids
    def _assignCentroids(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for pointIdx, samplePoint in enumerate(self.X):
            distancesToCentroids = []
            for _, centroidPoint in enumerate(centroids):
                distance = euclidean_distance(samplePoint, centroidPoint)
                distancesToCentroids.append(distance)
            closestDistance = np.argmin(distancesToCentroids)
            clusters[closestDistance].append(pointIdx)
        return clusters

    # Calculate new centroids as the mean of samples in each cluster
    def _updateCentroids(self, clusters, numOfFeatures):
        centroids = np.zeros((self.K, numOfFeatures))
        for clusterIdx, clusterPointsIndices in enumerate(clusters):
            clusterMean = np.sum(self.X[clusterPointsIndices], axis=0) / len(self.X[clusterPointsIndices])
            centroids[clusterIdx] = clusterMean
        return centroids

    # Check if the algorithm has converged (i.e., centroids have not changed)
    def _algoConvergence(self, prevCentroids, newCentroids):
        distances = [euclidean_distance(prevCentroids[i], newCentroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    # Plot the clusters and centroids
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
