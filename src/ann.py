import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# MSE Loss Function
def mean_square_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Abstract Base Class for layers
class Layer:
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, d_output):
        raise NotImplementedError

# Dense layer class
class Dense(Layer):
    def __init__(self, input_size, output_size, activation_function):
        # Constructor
        super().__init__()

        # Data dimension
        self.input_size = input_size

        # Units
        self.output_size = output_size

        # Activation Function
        self.activation_function = activation_function

        # Initialize weights and biases with random values
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        
        # Gradients for weight and bias updates
        self.d_weights = np.zeros((output_size, input_size))
        self.d_biases = np.zeros((output_size, 1))

    def forward(self, inputs):
        # Save input data
        self.inputs = inputs

        # Compute pre activation values of the neurons
        self.z = np.dot(self.weights, inputs) + self.biases
        
        # Applying activation function to each neuron values
        self.output = self.activation_function(self.z)

        return self.output

    def backward(self, d_output):
        # Calculate the gradient of the loss with respect to the pre-activation values
        d_activation = d_output * self.activation_derivative(self.z)

        # Calculate the gradient of the loss with respect to inputs of layer
        d_input = np.dot(self.weights.T, d_activation)

        # Calculate the gradient of the loss with respect to the weights of layer
        self.d_weights = np.dot(d_activation, self.inputs.T)

        # Calculate the gradient of the loss with respect to the biases of layer
        self.d_biases = np.sum(d_activation, axis=1, keepdims=True)
        
        return d_input

    def activation_derivative(self, x):
        # For this assignment, only implemented for sigmoid and relu
        if self.activation_function == sigmoid:
            # Define derivative of sigmoid
            return sigmoid(x) * (1 - sigmoid(x))
        elif self.activation_function == relu:
            # Define derivative of relu
            return (x > 0).astype(float)
        else:
            raise NotImplementedError("Activation function derivative not implemented.")

    def update_weights(self, learning_rate):
        # Stochastic Gradient Descent
        # Substract weights with product of learning rate and derivative weights
        self.weights -= learning_rate * self.d_weights

        # Substract biases with product of learning rate and derivative weights
        self.biases -= learning_rate * self.d_biases

# Sequential model class
class Sequential:
    def __init__(self):
        # Constructor
        self.layers = []

    def add(self, layer):
        # Add new layer
        self.layers.append(layer)

    def forward(self, inputs):
        # Iterate all layers, connecting neurons from layers to layers
        for layer in self.layers:
            # Pass output from current layer to the next layer
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, d_output):
        # Iterate all layers in reverse order
        for layer in reversed(self.layers):
            # Calculate gradients for current layer and pass to next layer
            d_output = layer.backward(d_output)

    def update_weights(self, learning_rate):
        # Iterate all layers
        for layer in self.layers:
            # Update weights using gradients computed and scales them with learning rate
            layer.update_weights(learning_rate)

    def fit(self, X, y, epochs=100, learning_rate=0.01, batch_size=32):
        # Train ANN
        for epoch in range(epochs):
            # Define total loss
            total_loss = 0

            # Define size of data
            num_samples = len(X)

            # Shuffle the data for each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for batch_start in range(0, num_samples, batch_size):
                # Create a batch
                batch_indices = indices[batch_start: batch_start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward and backward propagation for the current batch
                # Define batch loss
                batch_loss = 0

                # Iterate for all batch data
                for i in range(len(X_batch)):
                    # Define inputs
                    inputs = X_batch[i].reshape(-1, 1)

                    # Define target
                    target = y_batch[i]

                    # Forward propagation
                    prediction = self.forward(inputs)

                    # Compute loss
                    loss = mean_square_error(target, prediction)
                    batch_loss += loss

                    # Backward propagation
                    d_output = -(target - prediction)
                    self.backward(d_output)

                    # Weight update using SGD
                    self.update_weights(learning_rate)

                # Calculate average loss for the batch
                average_loss = batch_loss / len(X_batch)
                total_loss += average_loss

            # Calculate average loss for the epoch
            average_loss = total_loss / (num_samples // batch_size)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")
    
    def predict(self, X):
        # Define predictions
        predictions = []

        # Iterate all test data
        for i in range(len(X)):
            # Get features (inputs)
            inputs = X[i].reshape(-1, 1)

            # Predict
            prediction = self.forward(inputs)
            
            # Make prediction for binary classification task
            binary_prediction = (prediction >= 0.5).astype(int)

            # Add prediction
            predictions.append(binary_prediction.item())

        # Return predictions
        return np.array(predictions)