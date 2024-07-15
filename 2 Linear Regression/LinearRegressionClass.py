import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, numOfIterations=100):
        self.lr = lr  # Learning rate for gradient descent
        self.numOfIterations = numOfIterations  # Number of epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        numOfSamples, numOfFeatures = X.shape  # Number of samples and features
        self.weights = np.random.randn(numOfFeatures) * 0.01  # Initialize weights with small random values
        self.bias = 0  # Initialize bias with zero
        
        for epoch in range(self.numOfIterations):
            yhat = np.dot(X, self.weights) + self.bias  # Predicted values (numOfSamples, )
            
            # We take transpose to convert shape of X from (numOfSamples, numOfFeatures) to (numOfFeatures, numOfSamples) so it can be compatible with (yhat - y ) with shape of (numOfSamples, )
            dw = (1 / numOfSamples) * np.dot(X.T, yhat - y)  # Gradient of the loss with respect to weights (numOfFeatures, )
            db = (1 / numOfSamples) * np.sum(yhat - y)  # Gradient of the loss with respect to bias (scalar)

            self.weights = self.weights - self.lr * dw  # Update weights
            self.bias = self.bias - self.lr * db  # Update bias
            
            loss = self.MeanSquaredErrorLoss(y, yhat)
            print(f"Epoch {epoch+1}/{self.numOfIterations}, Loss: {loss}")
            
    def predict(self, X_test):
        predictions = np.dot(X_test, self.weights) + self.bias
        return predictions
    
    def MeanSquaredErrorLoss(self, y, yhat):
        loss = np.mean((y - yhat) ** 2)
        return loss
