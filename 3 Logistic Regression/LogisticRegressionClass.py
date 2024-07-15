import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.001, numOfIterations=1000):
        self.lr = lr  # Learning rate for gradient descent
        self.numOfIterations = numOfIterations  # Number of epochs
        self.weights = None
        self.bias = None
        
    # def _sigmoid(self, x):
    #     return 1/(1+np.exp(-x))
    
    def _sigmoid(self, x):
        # For elements in x that are negative (x < 0), it computes np.exp(x) / (1 + np.exp(x)). This approach avoids overflow that could occur when calculating np.exp(-x) directly for very large negative values of x
        # For x >= 0: Uses the standard sigmoid function 1 / (1 + np.exp(-x)). This is numerically stable for positive x.
        # For x < 0: Uses an equivalent form np.exp(x) / (1 + np.exp(x)). This form avoids computing np.exp(-x) directly when x is negative and large, which helps avoid overflow or underflow for maintaining numerical stability. 
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

        
    def fit(self, X, y):
        numOfSamples, numOfFeatures = X.shape  # Number of samples and features
        self.weights = np.random.randn(numOfFeatures) * 0.01  # Initialize weights with small random values
        self.bias = 0  # Initialize bias with zero
        
        for epoch in range(self.numOfIterations):
            linear_pred = np.dot(X, self.weights) + self.bias  # Predicted values (numOfSamples, )
            yhat = self._sigmoid(linear_pred)
            
            # We take transpose to convert shape of X from (numOfSamples, numOfFeatures) to (numOfFeatures, numOfSamples) so it can be compatible with (yhat - y ) with shape of (numOfSamples, )
            dw = (1 / numOfSamples) * np.dot(X.T, (yhat - y))  # Gradient of the loss with respect to weights (numOfFeatures, )
            db = (1 / numOfSamples) * np.sum(yhat - y)  # Gradient of the loss with respect to bias (scalar)

            self.weights = self.weights - self.lr * dw  # Update weights
            self.bias = self.bias - self.lr * db  # Update bias
            
            loss = self.BinaryCrossEntropyLoss(y, yhat)
            print(f"Epoch {epoch+1}/{self.numOfIterations}, Loss: {loss}")
            
    def predict(self, X_test):
        linear_predictions = np.dot(X_test, self.weights) + self.bias
        probabilities = self._sigmoid(linear_predictions)
        # If probability is greater than 0.5 then assign 1 and when it is less then assign 0
        predictions = [0 if pr <= 0.5 else 1 for pr in probabilities]
        return predictions
    
    def BinaryCrossEntropyLoss(self, y, yhat):
        yhat = np.clip(yhat, 1e-15, 1 - 1e-15)    # Clip probabilities to prevent log(0) and log(1)
        loss = np.mean(-(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))
        return loss
