import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

class Stochastic_GD_Algorithms:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = 100
        self.lmbda = 0.0001
        self.X = np.c_[np.ones((self.n,1)), self.x]
        self.XT_X = self.X.T @ self.X

    def learning_schedule(t):
        t0, t1 = 5, 50
        return t0/(t+t1)
    
    # Define the SGD function
    def Stochastic_gradient_descent(self, n_epochs, batch_size, reg, lmbda=1, momentum = False):
        n_batches = self.n // batch_size
        beta = np.random.randn(2,1)

        #parameters for momentum based SGD
        delta_momentum = 0.3
        change = 0.0

        for epoch in range(n_epochs):
            # Shuffle the data for each epoch
            permutation = np.random.permutation(self.n)
            X_shuffled = self.X[permutation]
            y_shuffled = self.y[permutation]

            for batch in range(n_batches):
                # Select a mini-batch
                start = batch * batch_size
                end = (batch + 1) * batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute the gradient for the mini-batch
                if reg == "OLS":
                    gradient = (2.0/batch_size)* X_batch.T @ ((X_batch @ beta)-y_batch)
                elif reg == "Ridge":
                    gradient = (2.0/batch_size)* X_batch.T @ ((X_batch @ beta)-y_batch)+2*lmbda*beta
                    
                eta = self.learning_schedule(epoch*n_batches+batch)

                #For comparing with momentum based Stochastic Gradient Descent
                if momentum == True:
                    new_change = eta*gradient+delta_momentum*change
                    beta -= new_change
                    change = new_change

                elif momentum == False:
                    beta -= eta*gradient

                if abs(gradient.all()) <= 10e-8:
                    break

        y_pred = self.X @ beta
        
        return y_pred, beta
    
    def Compare_plot(self, reg, y_pred, y_pred_m):    
        #Plotting the reg againt the gradient descent
        plt.plot(self.x, y_pred, "r-", label = "SGD")
        plt.plot(self.x, y_pred_m, "g-", label = "SGD with momentum")
        plt.plot(self.x, self.y ,'bo', label = "f(x)")
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(f'Stochastic gradient descent example for {reg} Regression')
        plt.legend()
        plt.show()

    def SGD_AutoGrad(self, n_epochs, batch_size, reg, lmbda=1.0, momentum = False):
        n_batches = self.n // batch_size
        beta = np.random.randn(2,1)
        
        #parameters for momentum based SGD
        delta_momentum = 0.3
        change = 0.0

        for epoch in range(n_epochs):
            # Shuffle the data for each epoch
            permutation = np.random.permutation(self.n)
            X_shuffled = self.X[permutation]
            y_shuffled = self.y[permutation]

            for batch in range(n_batches):
                # Select a mini-batch
                start = batch * batch_size
                end = (batch + 1) * batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                #Cost function OLS
                def Cost_OLS_momentum(beta):
                    residuals = X_batch @ beta - y_batch
                    cost = (1 / batch_size) * np.linalg.norm(residuals) ** 2
                    return cost
                
                def CostRidge_momentum(beta):
                    residuals = X_batch @ beta - y_batch
                    ridge_penalty = lmbda * np.linalg.norm(beta) ** 2
                    cost = (1 / batch_size) * (np.linalg.norm(residuals) ** 2 + ridge_penalty)
                    return cost
                    
                # Compute the gradient for the mini-batch
                if reg == "OLS":
                    training_gradient = grad(Cost_OLS_momentum)
                elif reg == "Ridge":
                    training_gradient = grad(CostRidge_momentum)
                    
                gradient = training_gradient(beta)
                eta = self.learning_schedule(epoch*n_batches+batch)

                #For comparing with momentum based Gradient Descent
                if momentum == True:
                    new_change = eta*gradient+delta_momentum*change
                    beta -= new_change
                    change = new_change

                elif momentum == False:
                    beta -= eta* gradient

                if abs(gradient.all()) <= 10e-8:
                    break

        y_pred = self.X @ beta
        
        return y_pred, beta
