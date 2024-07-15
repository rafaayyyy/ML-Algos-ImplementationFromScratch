from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from LogisticRegressionClass import LogisticRegression

# Load the dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Print dataset information
print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of target labels: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}\n")

# Print feature names and target names
# print("Feature names:")
# print(data.feature_names)
# print("\nTarget names:")
# print(data.target_names)
# print()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training and Testing Set Information:")
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}\n")

# Initialize and train the Decision Tree classifier
print("Training the Decision Tree classifier...\n")
clf = LogisticRegression(numOfIterations=1000)
clf.fit(X_train, y_train)

# Predict the labels for the test set
predictions = clf.predict(X_test)

print("Predicting the test set labels...\n")

# Calculate and print the accuracy of the predictions
acc = np.sum(y_test == predictions) / len(y_test)
print("Prediction Accuracy:")
print(f"Accuracy: {acc:.4f}")
