import numpy as np
from collections import Counter
from DecisionTreeClass import DecisionTree

# RandomForest class represents the random forest classifier
# Most of the default parameters will be used for Decision Trees Individual Classifiers
class RandomForest:
    def __init__(self, numOfTrees=10, minSamplesSplit=2, maxDepth=100, numOfFeatures=None):
        self.numOfTrees = numOfTrees  # Number of trees in the forest
        self.minSamplesSplit = minSamplesSplit  # Minimum samples required to split
        self.maxDepth = maxDepth  # Maximum depth of the tree
        self.numOfFeatures = numOfFeatures  # Number of features to consider for splitting
        self.trees = []  # List to store all the individual Decision Trees in the forest

    def fit(self, X, y):
        numOfSamples = X.shape[0]  # Number of samples in the dataset
        for _ in range(self.numOfTrees):
            baggingSampleIndices = np.random.choice(numOfSamples, numOfSamples, replace=True)  # Generate a bootstrap sample (bagging) with replacement
            tree = DecisionTree(minSamplesSplit=self.minSamplesSplit, maxDepth=self.maxDepth, numOfFeatures=self.numOfFeatures)
            # Fit the Decision Tree on the bootstrap sample
            tree.fit(X[baggingSampleIndices], y[baggingSampleIndices])
            self.trees.append(tree)

    def predict(self, X_test):
        
        predictions = np.array([tree.predict(X_test) for tree in self.trees])  # Get predictions from each tree in the forest
        treePredictions = np.swapaxes(predictions, 0, 1)  # Transpose the array to get predictions for each sample
        resultantPredictions = np.array([])

        # Find the most common occurrence (majority vote) for each sample's predictions
        for pred in treePredictions:
            mostCommonLabel = Counter(pred).most_common(1)[0][0]
            resultantPredictions = np.append(resultantPredictions, [mostCommonLabel])

        return resultantPredictions  # Final Result
