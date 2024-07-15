import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegressionClass import LinearRegression

def generate_dataset(num_features):
    if num_features == 1:
        X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    elif num_features == 2:
        X, y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=42)
    else:
        raise ValueError("Invalid number of features. Please enter 1 or 2.")
    return X, y

def visualize_dataset(X, y, num_features):
    if num_features == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
        plt.xlabel('Feature 1')
        plt.ylabel('Target')
        plt.title('Dataset with 1 Feature')
        plt.show()
    elif num_features == 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y, color='b', marker='o')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.set_title('Dataset with 2 Features')
        plt.show()
        
# Calculate mean squared error loss
def MeanSquaredErrorLoss(y_test, predictions):
    return np.mean((y_test - predictions)**2)

def train_and_predict(X_train, y_train, X_test, y_test, num_features, X):
    reg = LinearRegression(lr=0.1)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    loss = MeanSquaredErrorLoss(y_test, predictions)
    print(f"Mean Squared Error Loss: {loss}")
    
    # Plot predictions
    if num_features == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color='g', marker='o', label='Training data')
        plt.scatter(X_test, y_test, color='b', marker='o', label='Testing data')
        plt.plot(X_train, reg.predict(X_train), color='r', linewidth=2, label='Prediction')
        plt.xlabel('Feature 1')
        plt.ylabel('Target')
        plt.title('Linear Regression Prediction with 1 Feature')
        plt.legend()
        plt.show()
    elif num_features == 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='g', marker='o', label='Training data')
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='b', marker='o', label='Testing data')
        ax.scatter(X[:, 0], X[:, 1], reg.predict(X), color='r', marker='x', label='Prediction')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.set_title('Linear Regression Prediction with 2 Features')
        plt.legend()
        plt.show()

def main():
    # Ask user for the number of features
    num_features = int(input("Enter the number of features (1 or 2): "))
    
    X, y = generate_dataset(num_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Visualize dataset
    visualize_dataset(X, y, num_features)
    
    # Train and predict using linear regression
    train_and_predict(X_train, y_train, X_test, y_test, num_features, X)

if __name__ == "__main__":
    main()
