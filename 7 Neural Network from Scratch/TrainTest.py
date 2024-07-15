# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NeuralNetworkClass import NeuralNetwork

# %%
# Fetch the MNIST dataset
print("Fetching MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target'].astype(int)

# Print sizes and shapes before standardization
print(f"Before Standardization:")
print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}\n")

# %%
X = np.array(X)
y = np.array(y)

# %%
# Standardize the data
print("Standardizing data...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transpose the data arrays for compatibility with the NeuralNetwork class
print("Transposing data for NeuralNetwork compatibility...")
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

# Print sizes and shapes after transposing
print(f"After Transposing:")
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}\n")

# %%
# Initialize and train the neural network
print("Initializing and training NeuralNetwork...")
nn = NeuralNetwork(epochs = 1500, learningRate = 0.1)
nn.train(X_train, y_train)

# %%
# Make predictions on the test set
print("Making predictions on the test set...")
predictions = nn.predict(X_test)

# Function to calculate accuracy
def accuracy(predictions, y_test):
    return np.sum(predictions == y_test) / y_test.size

# Calculate and print the accuracy of the model on the test set
acc = accuracy(predictions, y_test)
print(f"\nTest set accuracy: {acc * 100:.4f}%")

# %%
# This function will display a picture from the test set along with the predicted and actual labels

def picturePredict(X_test, y_test, index):
    current_image = X_test[:, index].reshape(-1, 1)
    predicted_label = predictions[index]
    correct_label = y_test[index]
    
    current_image = current_image.reshape((28, 28)) * 255  # Reshape for plotting
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title(f'Predicted Label: {predicted_label}, Actual Label: {correct_label}')
    plt.show()

# %%
# unscale the test data
X_test_unscaled = scaler.inverse_transform(X_test.T)

# %%
# Visualize prediction for random images in the test set
randomIndices = np.random.choice(X_test.shape[1], 10, replace=False)
for index_to_visualize in randomIndices:
    try:
        picturePredict(X_test = X_test_unscaled.T,y_test = y_test, index=index_to_visualize)
    except IndexError as e:
        print(f"Error: {e}")

# %%
# This will visualize the prediction for the image at the specified index

index_to_visualize = 0
if index_to_visualize >= X_test.shape[1]:
    index_to_visualize = X_test.shape[1] - 1
try:
    picturePredict(X_test = X_test_unscaled.T,y_test = y_test, index=index_to_visualize)
except IndexError as e:
    print(f"Error: {e}")

# %%
# This will predict for a single image
index_to_predict = 0
X_single = X_test[:, index_to_predict].reshape(-1, 1)
prediction = nn.predict(X_single)
print(f"Predicted Label: {prediction}")
actual = y_test[index_to_predict]
print(f"Actual Label: {actual}")


