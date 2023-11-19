import math
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample

#Defining different activation function
def identity(X):
    return X

def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))

def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)

def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)

#Cost function to use
def CostOLS(target):

    def func(X):
        if target.shape[0] == 0:
            raise ValueError("Target array should not have zero elements.")

        return (1.0 / max(1, target.shape[0])) * np.sum((target - X) ** 2)

    return func

def RidgeCost(target):
    alpha = 0.1
    def func(X):
        if target.shape[0] == 0:
            raise ValueError("Target array should not have zero elements.")
        return np.mean((target - X) ** 2) + alpha * np.sum(np.square(X))
    return func

def CrossEntropy(y):

    def func(X):
        return -(1.0/y.size)*np.sum(y*np.log(X + 10e-10))

    return func

class Optimizer:
    def __init__(self, eta):
        self.eta = eta
    
    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass

class Constant(Optimizer):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass

def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return np.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)

class Neural_Network:
    def __init__(self,
                 dimensions: tuple[int],
                 activation_func: Callable = sigmoid,
                 output_func: Callable = lambda x: x,
                 cost_func: Callable = CostOLS,
                 seed: int = None
                 ):
        
        self.dimensions = dimensions
        self.activation_func = activation_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.update_weights = list()
        self.update_bias = list()
        self.input_matrices = list()
        self.output_matrices = list()
        self.classification = None

        self.reset_weights()
        self.problem_type()

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              optim: Optimizer,
              batches: int = 5,
              epochs: int = 100,
              lam: float = 0.1,
              X_val: np.ndarray = None,
              y_target: np.ndarray = None,
              ):
        
        if self.seed is not None:
            np.random.seed(self.seed)

        val_set = False
        if X_val is not None and y_target is not None:
            val_set = True

        
        # creating arrays for score metrics
        #MSE
        train_errors = np.empty(epochs)
        test_errors = np.empty(epochs)
        #R2 score
        train_r2scores = np.empty(epochs)
        test_r2scores = np.empty(epochs)
        train_accs = np.empty(epochs)
        test_accs = np.empty(epochs)

        self.optim_weight = list()
        self.optim_bias = list()

        batch_size = X.shape[0] // batches

        X, y = resample(X, y)

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(y)
        
        if val_set:
            cost_function_val = self.cost_func(y_target)
        
        # create optimizer for each weight matrix
        for i in range(len(self.weights)):
            self.optim_weight.append(copy(optim))
            self.optim_bias.append(copy(optim))

        for e in range(epochs):
            for i in range(batches):
                # allows for stochastic gradient descent
                if i == batches - 1:
                    # If the for loop has reached the last batch, take all thats left
                    X_batch = X[i * batch_size :, :]
                    y_batch = y[i * batch_size :, :]
                else:
                    X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                    y_batch = y[i * batch_size : (i + 1) * batch_size, :]

                self.forward(X_batch)
                self.backpropagate(X_batch, y_batch, lam)

            # reset optimizer for each epoch
            for optim in self.optim_weight:
                optim.reset()

            for optim in self.optim_bias:
                optim.reset()

            # computing performance metrics
            train_pred = self.predict(X)
            train_error = cost_function_train(train_pred)
            
            train_r2score = self.R2(train_pred, y)

            train_r2scores[e] = train_r2score
            train_errors[e] = train_error

            if val_set:
                test_pred = self.predict(X_val)
                target_error = cost_function_val(test_pred)
                test_errors[e] = target_error
                test_r2score = self.R2(test_pred, y)
                test_r2scores[e] = test_r2score

            if self.classification:
                train_acc = self.accuracy_score(self.predict(X), y)
                train_accs[e] = train_acc

                target_acc = self.accuracy_score(test_pred, y_target)
                test_accs[e] = target_acc

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors
        scores["train_r2scores"] = train_r2scores
        
        if val_set:
            scores["test_errors"] = test_errors
            scores["test_r2scores"] = test_r2scores

        if self.classification:
            scores["train_accs"] = train_accs
            scores["test_accs"] = test_accs

        return scores
    
    def predict(self, X: np.ndarray, *, threshold=0.5):
        predict = self.forward(X)

        if self.classification:
            return np.where(predict > threshold, 1, 0)
        else:
            return predict
        
    def reset_weights(self):
        #Resets the weigts for a new problem
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    def forward(self, X: np.ndarray):
        # reset matrices
        self.input_matrices = list()
        self.output_matrices = list()

        # if X is just a vector, make it into a matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Add a coloumn of zeros as the first coloumn of the design matrix, in order
        # to add bias to our data
        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        # a^0, the nodes in the input layer (one a^0 for each row in X - where the
        # exponent indicates layer number).
        input = X
        self.input_matrices.append(input)
        self.output_matrices.append(input)

        # The feed forward algorithm
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                output = input @ self.weights[i]
                self.output_matrices.append(output)
                input = self.activation_func(output)
                # bias column again added to the data here
                bias = np.ones((input.shape[0], 1)) * 0.01
                input = np.hstack([bias, input])
                self.input_matrices.append(input)
            else:
                try:
                    # the nodes in our output layers
                    output = input @ self.weights[i]
                    input = self.output_func(output)
                    self.input_matrices.append(input)
                    self.output_matrices.append(output)
                except Exception as OverflowError:
                    print(
                        "OverflowError in fit() in FFNN\nHOW TO DEBUG ERROR: Consider lowering your learning rate or scheduler specific parameters such as momentum, or check if your input values need scaling"
                    )

        return input
    

    def backpropagate(self, X, y, lam):
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.activation_func)

        for i in range(len(self.weights) - 1, -1, -1):
            # delta terms for output
            if i == len(self.weights) - 1:
                # for multi-class classification
                if (self.output_func.__name__ == "softmax"):
                    delta_matrix = self.input_matrices[i + 1] - y
                # for single class classification
                else:
                    cost_func_derivative = grad(self.cost_func(y))
                    delta_matrix = out_derivative(
                        self.output_matrices[i + 1]
                    ) * cost_func_derivative(self.input_matrices[i + 1])

            # delta terms for hidden layer
            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.output_matrices[i + 1])

            # calculate gradient
            gradient_weights = self.input_matrices[i][:, 1:].T @ delta_matrix
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(
                1, delta_matrix.shape[1]
            )

            # regularization term
            gradient_weights += self.weights[i][1:, :] * lam

            # use optimizer
            update_matrix = np.vstack(
                [
                    self.optim_bias[i].update_change(gradient_bias),
                    self.optim_weight[i].update_change(gradient_weights),
                ]
            )

            # update weights and bias
            self.weights[i] -= update_matrix

    def accuracy_score(self, prediction: np.ndarray, target: np.ndarray):
        assert prediction.size == target.size
        return np.average((target == prediction))
    
    def R2(self, prediction, target):
        return 1 - np.sum((target - prediction)** 2)/np.sum((target - np.mean(target))**2)
    
    def problem_type(self):
        #Should the network work as a classifier
        #or as a regressor
        self.classification = False
        if (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            self.classification = True



