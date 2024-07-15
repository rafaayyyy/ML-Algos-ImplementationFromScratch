import numpy as np

class NeuralNetwork:
    def __init__(self, epochs=500, learningRate=0.1):
        # Initialize weights and biases for layer 1 (W1, B1)
        # Weights (W1) shape: (10, 784)
        # Biases (B1) shape: (10, 1)
        self.__weightsLayer1 = np.random.rand(10, 784) - 0.5
        self.__biasLayer1 = np.random.rand(10, 1) - 0.5
        # Initialize weights and biases for layer 2 (W2, B2)
        # Weights (W2) shape: (10, 10)
        # Biases (B2) shape: (10, 1)
        self.__weightsLayer2 = np.random.rand(10, 10) - 0.5
        self.__biasLayer2 = np.random.rand(10, 1) - 0.5
        self.__epochs = epochs
        self.__learningRate = learningRate
        
    def __softmax(self, Z):
        # Z shape: (10, m) where m is the number of samples
        # Apply softmax activation function to compute output probabilities
        # A shape: (10, m)
        # axis = 0 is for moving downwards meaning 1 sample's probabilities are in the same column
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        return A
    
    def __ReLU(self, Z):
        # Z shape: (10, m) where m is the number of samples
        # Apply ReLU activation function to introduce non-linearity
        # A shape: (10, m)
        return np.maximum(0, Z)
    
    def __derivativeReLU(self, Z):
        # Z shape: (10, m) where m is the number of samples
        # Derivative of ReLU function used for backpropagation
        # dReLU shape: (10, m)
        return Z > 0
    
    def __oneHotOutputLabels(self, Y):
        # Y shape: (m,) where m is the number of samples
        # Convert labels to one-hot encoding
        encodedLabels = np.zeros((Y.size, Y.max() + 1))  # size here is (m, 10) where m is the number of samples
        encodedLabels[np.arange(Y.size), Y] = 1
        encodedLabels = encodedLabels.T  # we take the transpose to get the shape (10, m)
        # encodedLabels shape: (10, m)
        return encodedLabels
    
    def __forward(self, X):
        # X shape: (784, m) where m is the number of samples
        # Forward pass: compute activations for both layers
        
        # Linear transformation for layer 1: L1 = W1 * X + B1
        # L1 shape: (10, m)
        self.__inputs = X
        self.__linearLayer1 = np.dot(self.__weightsLayer1, X) + self.__biasLayer1
        # Activation for layer 1: A1 = ReLU(L1)
        # A1 shape: (10, m)
        self.__activatedLayer1 = self.__ReLU(self.__linearLayer1)
        
        # Linear transformation for layer 2: L2 = W2 * A1 + B2
        # L2 shape: (10, m)
        self.__linearLayer2 = np.dot(self.__weightsLayer2, self.__activatedLayer1) + self.__biasLayer2
        # Activation for layer 2: A2 = softmax(L2)
        # A2 shape: (10, m)
        self.__activatedLayer2 = self.__softmax(self.__linearLayer2)
        
        # This returns the probabilities of each class for all samples
        return self.__activatedLayer2
    
    def __backward(self, Y):
        # Y shape: (m,) where m is the number of samples
        # Backward pass: compute gradients and propagate error backward
        
        # Convert labels to one-hot encoding
        self.__Y = self.__oneHotOutputLabels(Y)
        m = Y.shape[0]  # Number of samples
        
        # Error at output layer: dLoss/dL2 = A2 - Y (Combined derivative of softmax and cross-entropy loss function)
        # A2 shape: (10, m)
        # Y shape: (10, m)
        # dLoss/dL2 shape: (10, m)
        self.__errorLayer2 = self.__activatedLayer2 - self.__Y
        # Gradient for weights of layer 2: dLoss/dW2
        # dLoss/dW2 = dLoss/dL2 * dL2/dW2 = (A2 - Y) * A1  (divide by number of samples for average gradient)
        # dLoss/dW2 shape: (10, 10)
        self.__gradientWeightsLayer2 = 1 / m * np.dot(self.__errorLayer2, self.__activatedLayer1.T)
        # Gradient for biases of layer 2: dLoss/dB2
        # dLoss/dB2 = dLoss/dL2 * dL2/dB2 = (A2 - Y) * 1  (divide by number of samples for average gradient)
        # dLoss/dB2 shape: (10, 1)
        self.__gradientBiasLayer2 = 1 / m * np.sum(self.__errorLayer2, axis=1, keepdims=True)
        
        # Propagate error to layer 1: dLoss/dA1
        # dLoss/dA1 = dLoss/dL1 * dL1/dA1 = (dLoss/dL2 * W2) * dReLU(A1) (dReLU(A1  ) is the d(A1)/dL1)
        # dLoss/dA1 shape: (10, m)
        self.__errorLayer1 = np.dot(self.__weightsLayer2.T, self.__errorLayer2) * self.__derivativeReLU(self.__activatedLayer1)
        # Gradient for weights of layer 1: dLoss/dW1
        # dLoss/dW1 = dLoss/dL1 * dL1/dW1 = (dLoss/dA1 * X) where X = inputs (divide by number of samples for average gradient)
        # dLoss/dW1 shape: (10, 784)
        self.__gradientWeightsLayer1 = 1 / m * np.dot(self.__errorLayer1, self.__inputs.T)
        # Gradient for biases of layer 1: dLoss/dB1
        # dLoss/dB1 = dLoss/dL1 * dL1/dB1 = (dLoss/dA1 * 1)  (divide by number of samples for average gradient)
        # dLoss/dB1 shape: (10, 1)
        self.__gradientBiasLayer1 = 1 / m * np.sum(self.__errorLayer1, axis=1, keepdims=True)
        
    def __updateParams(self):
        # Update parameters using the calculated gradients and learning rate
        
        # Update weights and biases for layer 1: W1, B1
        # W1 shape: (10, 784)
        # B1 shape: (10, 1)
        self.__weightsLayer1 -= self.__learningRate * self.__gradientWeightsLayer1
        self.__biasLayer1 -= self.__learningRate * self.__gradientBiasLayer1
        # Update weights and biases for layer 2: W2, B2
        # W2 shape: (10, 10)
        # B2 shape: (10, 1)
        self.__weightsLayer2 -= self.__learningRate * self.__gradientWeightsLayer2
        self.__biasLayer2 -= self.__learningRate * self.__gradientBiasLayer2
        
    def __getPredictions(self, probs):
        # probs shape: (10, m) where m is the number of samples
        # Get predicted labels from output probabilities
        # predictions shape: (m,)
        return np.argmax(probs, axis=0)

    def __accuracy(self, predictions, y):
        # predictions shape: (m,)
        # y_test shape: (m,)
        # Calculate accuracy of predictions
        return np.sum(predictions == y) / y.size
    
    def __categoricalCrossEntropyLoss(self, probs):
        # probs shape: (10, m) where m is the number of samples
        # Compute categorical cross-entropy loss
        
        m = self.__Y.shape[1]  # Number of samples
        # Clip probabilities to avoid log(0) and division by zero errors
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        # Compute the log probabilities
        log_probs = np.sum(self.__Y * -np.log(probs), axis=0)
        # Compute the loss
        loss = np.sum(log_probs) / m
        return loss
    
    def train(self, X, Y):
        # X shape: (784, m (56000 if 80:20 train-test split)) (number of training samples)
        # Y shape: (m (56000 if 80:20 train-test split),)
        # Train the neural network using forward and backward passes
        
        for epoch in range(self.__epochs + 1):
            # Forward pass: compute probabilities
            # probabilities shape: (10, m (56000 if 80:20 train-test split))
            probabilities = self.__forward(X)
            # Backward pass: compute gradients
            self.__backward(Y)
            # Update parameters
            self.__updateParams()
            # Compute loss
            loss = self.__categoricalCrossEntropyLoss(probabilities)
            # Print loss and accuracy at intervals
            if epoch % 50 == 0 or epoch == self.__epochs:
                predictedLabels = self.__getPredictions(probabilities)
                acc = self.__accuracy(predictedLabels, Y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc * 100:.4f}%")
                
    def predict(self, X_test):
        # X_test shape: (784, test-m (14000 if 80:20 train-test split))
        # Predict labels for test data
        
        # probabilities shape: (10, test-m (14000 if 80:20 train-test split))
        probabilities = self.__forward(X_test)
        # predictedLabels shape: (test-m (14000 if 80:20 train-test split),)
        self.predictedLabels = self.__getPredictions(probabilities)
        return self.predictedLabels

