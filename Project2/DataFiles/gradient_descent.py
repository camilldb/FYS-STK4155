import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

class Gradient_Descent_Algorithms:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = 100
        self.X = np.c_[np.ones((self.n,1)), self.x]
        self.XT_X = self.X.T @ self.X

    def GradientDescent(self, reg, lmbda=1.0, momentum = False):
        beta = np.random.randn(2,1)
        #Parameters for momentum based GD
        change = 0.0
        delta_momentum = 0.3
        
        for i in range(self.n):
            if reg == "OLS":
                gradient = (2.0/self.n)*self.X.T @ (self.X @ beta-self.y)
                # Hessian matrix
                H = (2.0/self.n)* self.XT_X
                EigValues, _ = np.linalg.eig(H)


            elif reg == "Ridge":
                gradient = (2.0/self.n)*self.X.T @ (self.X @ beta-self.y)+2*lmbda*beta
                #Hessian matrix
                H = (2.0/self.n)*self.XT_X+2*lmbda*np.eye(self.XT_X.shape[0])
                #Get the eigenvalues
                EigValues, _ = np.linalg.eig(H)

            #Learning parameter
            eta = 1.0/np.max(EigValues)
            
            #For comparing with momentum based Gradient Descent
            if momentum == False:
                beta -= eta*gradient

            elif momentum == True:
                new_change = eta*gradient+delta_momentum*change
                beta -= new_change
                change = new_change
            
            if abs(gradient.all()) <= 10e-8:
                break

        y_pred = self.X @ beta

        return y_pred, beta
    
    def Compare_plot(self, reg, y_pred, y_pred_m):
        plt.plot(self.x, y_pred, "r-", label = "GD")
        plt.plot(self.x, y_pred_m, "g-", label = "GD with momentum")
        plt.plot(self.x, self.y ,'bo', label = "f(x)")
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(f'Gradient descent example for {reg} Regression')
        plt.legend()
        plt.show()

    def CostOLS(self, beta):
        residuals = self.X @ beta - self.y
        cost = (1 / self.n) * np.linalg.norm(residuals) ** 2
        return cost
    
    def CostRidge(self, beta):
        residuals = self.X @ beta - self.y
        ridge_penalty = self.lmbda * np.linalg.norm(beta) ** 2
        cost = (1 / self.n) * (np.linalg.norm(residuals) ** 2 + ridge_penalty)
        return cost
    
    def GradientDescent_AutoGrad(self, reg, lmbda=1.0, momentum = False):
        #Guess the beta
        beta = np.random.randn(2,1)

        #Parameters for momentum based GD
        change = 0.0
        delta_momentum = 0.3

        for i in range(self.n):
            if reg == "OLS":
                # Hessian matrix
                H = (2.0/self.n)* self.XT_X
                # Get the eigenvalues
                EigValues, _ = np.linalg.eig(H)
                training_gradient = grad(self.CostOLS)

            elif reg == "Ridge":
                #Hessian matrix
                H = (2.0/self.n)*self.XT_X+2*lmbda*np.eye(self.XT_X.shape[0])
                #Get the eigenvalues
                EigValues, _ = np.linalg.eig(H)
                training_gradient = grad(self.CostRidge)

            #learning rate
            eta = 1.0/np.max(EigValues)
            gradient = training_gradient(beta)

            #For comparing with momentum based Gradient Descent
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