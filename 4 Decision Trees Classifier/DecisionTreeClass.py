import numpy as np
from collections import Counter

# Node class represents a node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Feature index used for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value if it's a leaf node (class label)
        
    def isLeafNode(self):
        # Returns True if the node is a leaf node
        return self.value is not None
        
# DecisionTree class represents the decision tree classifier
class DecisionTree:
    def __init__(self, minSamplesSplit = 2, maxDepth = 100, numOfFeatures = None):
        self.minSamplesSplit = minSamplesSplit  # Minimum samples required to split
        self.maxDepth = maxDepth  # Maximum depth of the tree
        self.numOfFeatures = numOfFeatures  # Number of features to consider for splitting
        self.root = None  # Root node of the tree
    
    def fit(self, X, y):
        if not self.numOfFeatures:
            self.numOfFeatures = X.shape[1]  # Set number of features to all features if not specified
        else:
            self.numOfFeatures = min(self.numOfFeatures, X.shape[1])  # Use specified number of features or all features, whichever is smaller
        self.root = self._growTreeFromNodes(X, y)  # Start growing the tree from the root
        
    def _growTreeFromNodes(self, X, y, depth = 0):
        numOfSamples = X.shape[0]  # Number of samples in the current node
        numOfFeatures = X.shape[1]  # Number of features in the dataset
        numOfPossibleLabels = len(np.unique(y))  # Number of unique class labels in the current node
        
        # Check for Stopping Criteria
        if(numOfPossibleLabels == 1 or depth >= self.maxDepth or numOfSamples < self.minSamplesSplit):
            temp = Counter(y)  # Count occurrences of each class label
            leafNodeValue = temp.most_common(1)[0][0]  # Get the most common class label
            return Node(value= leafNodeValue)  # Return a leaf node with the most common class label
        
        # Choose random features for splitting criteria
        selectedRandomFeaturesIndices = np.random.choice(numOfFeatures, self.numOfFeatures, replace = False)  # Select random unique features
        
        # Find the best split
        bestFeatureForSplit, bestThresholdForSplit = self._bestSplittingCriteria(X, y, selectedRandomFeaturesIndices)  # Determine the best feature and threshold for splitting
        
        # Create child nodes
        leftBranchIndices, rightBranchIndices = self._splitIntoBranches(X[:, bestFeatureForSplit], bestThresholdForSplit)
        leftChild = self._growTreeFromNodes(X[leftBranchIndices], y[leftBranchIndices], depth = depth + 1)  # Recursively grow the left child node
        rightChild = self._growTreeFromNodes(X[rightBranchIndices], y[rightBranchIndices], depth = depth + 1)  # Recursively grow the right child node
        return Node(bestFeatureForSplit, bestThresholdForSplit, leftChild, rightChild)  # Return the current decision node with the best feature, threshold, and child nodes

        
    def _bestSplittingCriteria(self, X, y, featuresIndices):
        bestInformationGain = -1
        bestSplittingFeature = None
        bestSplittingThreshold = None
        
        for feature in featuresIndices:
            X_Column = X[:, feature]
            possibleThresholds = np.unique(X_Column)  # Possible thresholds are unique values in the feature column
            
            for threshold in possibleThresholds:
                informationGainVal = self._informationGain(y, X_Column, threshold)
                
                if informationGainVal > bestInformationGain:
                    bestInformationGain = informationGainVal
                    bestSplittingFeature = feature
                    bestSplittingThreshold = threshold
                    
        return bestSplittingFeature, bestSplittingThreshold  # Returns best splitting criterion
    
    def _informationGain(self, y, X_Column, threshold):
        parentEntropy = self._entropy(y)  # Calculate the entropy of the parent node
        
        leftIndices, rightIndices = self._splitIntoBranches(X_Column, threshold)  # Split data into left and right branches based on the threshold
        
        if len(leftIndices) == 0 or len(rightIndices) == 0:
            return 0  # Return 0 if there is no valid split (either branch has no samples)
        
        totalSamples = len(y)  # Total number of samples
        leftBranchSamplesCount = len(leftIndices)
        rightBranchSamplesCount = len(rightIndices)
        leftBranchEntropy = self._entropy(y[leftIndices])
        rightBranchEntropy = self._entropy(y[rightIndices])
        
        # Calculate weighted entropy of the children
        childrenWeightedEntropy = (leftBranchSamplesCount / totalSamples) * leftBranchEntropy + \
                                  (rightBranchSamplesCount / totalSamples) * rightBranchEntropy

        informationGain = parentEntropy - childrenWeightedEntropy  # Calculate information gain by subtracting children's weighted entropy from parent's entropy
        return informationGain  # Return the information gain

    
    def _entropy(self, y):
        hist = np.bincount(y)  # Count occurrences of each class label
        probs = hist / len(y)  # Calculate probabilities
        return -np.sum([p * np.log(p) for p in probs if p>0])  # Calculate entropy
        
    def _splitIntoBranches(self, X_Column, splitThreshold):
        leftIndices = np.argwhere(X_Column <= splitThreshold).flatten()  # Indices of samples in the left branch
        rightIndices = np.argwhere(X_Column > splitThreshold).flatten()  # Indices of samples in the right branch
        return leftIndices, rightIndices
        
    def predict(self, X_test):
        # Predict class labels for the test set
        predictions = np.array([self._traverseTree(x, self.root) for x in X_test])
        return predictions
    
    def _traverseTree(self, x, node):
        # Traverse the tree to make a prediction for a single sample
        if node.isLeafNode():
            return node.value  # Return the value if it's a leaf node and class prediction will be returned 

        # Traverse the decision nodes
        if x[node.feature] <= node.threshold:
            return self._traverseTree(x, node.left)  # Traverse towards left subtree
        else:
            return self._traverseTree(x, node.right)  # Traverse towards right subtree
