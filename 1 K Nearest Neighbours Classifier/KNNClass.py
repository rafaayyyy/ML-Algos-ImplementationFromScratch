import numpy as np
from collections import Counter

# Euclidean distance between two points
def euclideanDistance(arr1, arr2):
  dist_arr = np.square(arr1 - arr2)
  distance = np.sqrt(np.sum(dist_arr))
  return distance

# Euclidean distance between a point and all points in an array
def combinedEuclideanDistances(arrays, point):
  dist_arr = np.square(arrays - point)
  distance = np.sqrt(np.sum(dist_arr, axis=1))
  return distance

class KNN():
  def __init__(self, k = 5):
    self.k = k

  def fit(self, data, targets):
    self.X_train = data
    self.y_train = targets

  def _predictSinglePoint(self, point):
    # distances = []
    # for idx, train_point in enumerate(self.X_train):
    #   dist = euclideanDistance(point, train_point)
    #   distances.append(dist)

    # distances = [euclideanDistance(point, train_point) for train_point in self.X_train]

    distances = combinedEuclideanDistances(self.X_train, point)

    sorted_array = np.argsort(distances)[:self.k]

    nearest_k_labels =  [self.y_train[i] for i in sorted_array]

    yhat = Counter(nearest_k_labels).most_common(1)
    return yhat[0][0]

  def predict(self, X_test):
    predictions = [self._predictSinglePoint(testSample) for testSample in X_test]
    return predictions
  
import numpy as np
from collections import Counter

# Function to calculate the Euclidean distance between two points
def euclideanDistance(arr1, arr2):
  # Calculate the squared difference, sum it, and take the square root
  dist_arr = np.square(arr1 - arr2)
  distance = np.sqrt(np.sum(dist_arr))
  return distance

# Function to calculate the Euclidean distance between a point and all points in an array
def combinedEuclideanDistances(arrays, point):
  # Calculate squared difference for each point, sum across features, and take the square root
  dist_arr = np.square(arrays - point)
  distance = np.sqrt(np.sum(dist_arr, axis=1))
  return distance

class KNN():
  def __init__(self, k = 5):
    # Initialize with k, the number of neighbors to consider
    self.k = k

  def fit(self, data, targets):
    # Store the training data and labels
    self.X_train = data
    self.y_train = targets

  def _predictSinglePoint(self, point):
    # distances = []
    # for idx, train_point in enumerate(self.X_train):
    #   dist = euclideanDistance(point, train_point)
    #   distances.append(dist)

    distances = [euclideanDistance(point, trainPoint) for trainPoint in self.X_train]
    
    # Calculate distances from the point to all training data
    # distances = combinedEuclideanDistances(self.X_train, point)

    # Get indices of the k nearest neighbors
    nearestNeighborIndices = np.argsort(distances)[:self.k]

    # Retrieve the labels of the k nearest neighbors
    nearestNeighborLabels = [self.y_train[i] for i in nearestNeighborIndices]

    # Determine the most common label among the nearest neighbors
    yhat = Counter(nearestNeighborLabels).most_common(1)
    return yhat[0][0]

  def predict(self, X_test):
    # Predict the label for each test point
    predictions = [self._predictSinglePoint(testPoint) for testPoint in X_test]
    return predictions