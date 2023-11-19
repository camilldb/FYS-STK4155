import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

class TuningLearningRate():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.X = np.c_[np.ones((self.n,1)), self.x]
        self.XT_X = self.X.T @ self.X
        #Number of datapoints
        self.n = 100
        #Value for learning rate
        self.eta = 0.01

    def tuning_learning_rate(self, reg, grad_type, tuning_type,
                         n_epochs = 50, batch_size = 5, 
                         lmbda = 1.0, momentum = False):
        #Guess for unknown parameter beta
        beta = np.random.randn(2, 1)
        #Parameters for momentum based (stochastic) gradient descent
        change = 0.0
        delta_momentum = 0.3

        #Including AdaGrad parameter to avoid possible division by zero
        delta = 1e-8

        if tuning_type == "RMSProp":
            # Value for parameter rho
            rho = 0.99

        elif tuning_type == "ADAM":
            # Value for parameters beta1 and beta2
            rho1 = 0.9
            rho2 = 0.999
            i = 0

        if grad_type == "GD":
            if tuning_type == "AdaGrad" or tuning_type == "RMSProp":
                Giter = 0.0
            elif tuning_type == "ADAM":
                first_moment = 0.0
                second_moment = 0.0

            for iter in range(self.n):
                if reg == "OLS":
                    gradient = (2.0/self.n)*self.X.T @ (self.X @ beta-self.y)

                elif reg == "Ridge":
                    gradient = (2.0/self.n)*self.X.T @ (self.X @ beta-self.y)+2*lmbda*beta
                
                if tuning_type == "AdaGrad":
                    Giter += gradient*gradient
                    update = gradient*self.eta/(delta+np.sqrt(Giter))
                    beta -= update

                elif tuning_type == "RMSProp":
                    # Scaling with rho the new and the previous results
                    Giter = (rho*Giter+(1-rho)*gradient*gradient)
                    # Taking the diagonal only and inverting
                    update = gradient*self.eta/(delta+np.sqrt(Giter))
                    # Hadamard product
                    beta -= update

                elif tuning_type == "ADAM":
                    i += 1
                    # Computing moments first
                    first_moment = rho1*first_moment + (1-rho1)*gradient
                    second_moment = rho2*second_moment+(1-rho2)*gradient*gradient
                    first_term = first_moment/(1.0-rho1**i)
                    second_term = second_moment/(1.0-rho2**i)
                    # Scaling with rho the new and the previous results
                    update = self.eta*first_term/(np.sqrt(second_term)+delta)
                    beta -= update
            
                #Momentum or not
                if momentum == True:
                    new_change = self.eta*gradient+delta_momentum*change
                    beta -= new_change
                    change = new_change
                elif momentum == False:
                    beta -= self.eta*gradient
                
                if abs(gradient.all()) <= 10e-8:
                    break

        elif grad_type == "SGD":
            n_batches = self.n//batch_size #number of minibatches

            for epoch in range(n_epochs):   
                if tuning_type == "AdaGrad" or tuning_type == "RMSProp":
                    Giter = 0.0
                elif tuning_type == "ADAM":
                    first_moment = 0.0
                    second_moment = 0.0
                    i += 1
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

                    #eta = learning_schedule(epoch*n_batches+batch)
                    
                    if tuning_type == "AdaGrad":
                        Giter += gradient*gradient
                        update = gradient*self.eta/(delta+np.sqrt(Giter))
                        beta -= update
                    elif tuning_type == "RMSProp":
                        # Scaling with rho the new and the previous results
                        Giter = (rho*Giter+(1-rho)*gradient*gradient)
                        # Taking the diagonal only and inverting
                        update = gradient*self.eta/(delta+np.sqrt(Giter))
                        # Hadamard product
                        beta -= update
                    elif tuning_type == "ADAM":
                        # Computing moments first
                        first_moment = rho1*first_moment + (1-rho1)*gradient
                        second_moment = rho2*second_moment+(1-rho2)*gradient*gradient
                        first_term = first_moment/(1.0-rho1**i)
                        second_term = second_moment/(1.0-rho2**i)
                        # Scaling with rho the new and the previous results
                        update = self.eta*first_term/(np.sqrt(second_term)+delta)
                        beta -= update

                    #Momentum or not
                    if momentum == True:
                        new_change = self.eta*gradient+delta_momentum*change
                        beta -= new_change
                        change = new_change
                    elif momentum == False:
                        beta -= self.eta*gradient
                    
                    if abs(gradient.all()) <= 10e-8:
                        break

        y_pred = self.X @ beta

        return y_pred, beta
    
    def TLR_autograd(self, reg, grad_type, tuning_type,
                         n_epochs = 50, batch_size = 5, 
                         lmbda = 1.0, momentum = False):
        #Guess for unknown parameter beta
        beta = np.random.randn(2, 1)
        #Parameters for momentum based (stochastic) gradient descent
        change = 0.0
        delta_momentum = 0.3

        #Including AdaGrad parameter to avoid possible division by zero
        delta = 1e-8

        if tuning_type == "RMSProp":
            # Value for parameter rho
            rho = 0.99

        elif tuning_type == "ADAM":
            # Value for parameters beta1 and beta2
            rho1 = 0.9
            rho2 = 0.999
            i = 0

        if grad_type == "GD":
            if tuning_type == "AdaGrad" or tuning_type == "RMSProp":
                Giter = 0.0
            elif tuning_type == "ADAM":
                first_moment = 0.0
                second_moment = 0.0

            for iter in range(self.n):
                if reg == "OLS":
                    training_gradient = grad(self.CostOLS)

                elif reg == "Ridge":
                    training_gradient = grad(self.CostRidge)

                gradient = training_gradient(beta)
                
                if tuning_type == "AdaGrad":
                    Giter += gradient*gradient
                    update = gradient*self.eta/(delta+np.sqrt(Giter))
                    beta -= update

                elif tuning_type == "RMSProp":
                    # Scaling with rho the new and the previous results
                    Giter = (rho*Giter+(1-rho)*gradient*gradient)
                    # Taking the diagonal only and inverting
                    update = gradient*self.eta/(delta+np.sqrt(Giter))
                    # Hadamard product
                    beta -= update

                elif tuning_type == "ADAM":
                    i += 1
                    # Computing moments first
                    first_moment = rho1*first_moment + (1-rho1)*gradient
                    second_moment = rho2*second_moment+(1-rho2)*gradient*gradient
                    first_term = first_moment/(1.0-rho1**i)
                    second_term = second_moment/(1.0-rho2**i)
                    # Scaling with rho the new and the previous results
                    update = self.eta*first_term/(np.sqrt(second_term)+delta)
                    beta -= update
                
                #Momentum or not
                if momentum == True:
                    new_change = self.eta*gradient+delta_momentum*change
                    beta -= new_change
                    change = new_change
                elif momentum == False:
                    beta -= self.eta*gradient
                
                if abs(gradient.all()) <= 10e-8:
                    break

        elif grad_type == "SGD":
            n_batches = self.n//batch_size #number of minibatches

            for epoch in range(n_epochs):   
                if tuning_type == "AdaGrad" or tuning_type == "RMSProp":
                    Giter = 0.0
                elif tuning_type == "ADAM":
                    first_moment = 0.0
                    second_moment = 0.0
                    i += 1
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
                    #Cost function OLS
                    def Cost_OLS_momentum(beta):
                        residuals = X_batch @ beta - y_batch
                        cost = (1 / batch_size) * np.linalg.norm(residuals) ** 2
                        return cost
                    #Cost function for Ridge 
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

                    #eta = learning_schedule(epoch*n_batches+batch)
                    
                    if tuning_type == "AdaGrad":
                        Giter += gradient*gradient
                        update = gradient*self.eta/(delta+np.sqrt(Giter))
                        beta -= update
                    elif tuning_type == "RMSProp":
                        # Scaling with rho the new and the previous results
                        Giter = (rho*Giter+(1-rho)*gradient*gradient)
                        # Taking the diagonal only and inverting
                        update = gradient*self.eta/(delta+np.sqrt(Giter))
                        # Hadamard product
                        beta -= update
                    elif tuning_type == "ADAM":
                        # Computing moments first
                        first_moment = rho1*first_moment + (1-rho1)*gradient
                        second_moment = rho2*second_moment+(1-rho2)*gradient*gradient
                        first_term = first_moment/(1.0-rho1**i)
                        second_term = second_moment/(1.0-rho2**i)
                        # Scaling with rho the new and the previous results
                        update = self.eta*first_term/(np.sqrt(second_term)+delta)
                        beta -= update

                    #Momentum or not
                    if momentum == True:
                        new_change = self.eta*gradient+delta_momentum*change
                        beta -= new_change
                        change = new_change
                    elif momentum == False:
                        beta -= self.eta*gradient
                    
                    if abs(gradient.all()) <= 10e-8:
                        break

        y_pred = self.X @ beta

        return y_pred, beta
    
    def Compare_plot(self, tuning_type, grad_type, reg, y_pred, y_pred_m):    
        #Plotting the reg againt the gradient descent
        plt.plot(self.x, y_pred, "r-", label = f"{grad_type}")
        plt.plot(self.x, y_pred_m, "g-", label = f"{grad_type} with momentum")
        plt.plot(self.x, self.y ,'bo', label = "f(x)")
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(f'{grad_type} example for {reg} Regression using {tuning_type} for tuning the learning rate')
        plt.legend()
        plt.show()

    