import numpy as np

class NeuralNetwork:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.W1 = np.random.randn(2, 2)  # Input to hidden
        self.W2 = np.random.randn(2, 1)  # Hidden to output

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        return self.sigmoid(self.z2)

    def backward(self, X, y, output):
        # Output layer error
        error2 = (output - y)
        delta2 = error2 * self.sigmoid(output, deriv=True)

        # Hidden layer error
        error1 = delta2.dot(self.W2.T)
        delta1 = error1 * self.sigmoid(self.a1, deriv=True)

        # Weight updates
        self.W2 -= self.alpha * self.a1.T.dot(delta2)
        self.W1 -= self.alpha * X.T.dot(delta1)

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [0], [0], [1]])

nn = NeuralNetwork(alpha=0.5)
nn.train(X, y)
print("AND Gate Prediction:")
print(np.round(nn.forward(X), 2))
