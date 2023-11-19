import numpy as np

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        for i in range(self.max_iters):
            prev_weights = np.copy(self.weights)
            idx = np.random.permutation(len(X))
            
            for j in idx:
                y_pred = self.sigmoid(np.dot(X[j], self.weights))
                error = y[j] - y_pred
                gradient = error * X[j]
                self.weights += self.learning_rate * gradient
            
            # Check convergence
            if np.linalg.norm(self.weights - prev_weights) < self.tol:
                break

    def predict(self, X):
        if self.weights is None:
            raise Exception("Model not trained yet. Please call fit() before predict().")

        y_pred = np.dot(X, self.weights)
        return np.round(self.sigmoid(y_pred))

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)