import numpy as np
from random import random, seed
import matplotlib.pyplot as plt
from math import exp, sqrt
from sklearn.model_selection import train_test_split
from gradient_descent import Gradient_Descent_Algorithms
from stochastic_gradient_descent import Stochastic_GD_Algorithms
from tuning_learning_rate import TuningLearningRate
from neural_network import Neural_Network, Optimizer, Constant
from neural_network import CostOLS, CrossEntropy, RidgeCost, sigmoid, RELU, LRELU
from logistic_regression import LogisticRegressionSGD

#a)
#simple function
def f(x):
    return 4.0 + x * 3.0 + x**2.0

n = 100
x = 2*np.random.rand(n,1)
y = f(x)+np.random.rand(n,1)

#OLS
X = np.c_[np.ones((n,1)), x]
XT_X = X.T @ X
beta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
print("Own inversion\n")
print(beta_linreg)

#Ridge
l = 0.0001
I = np.eye(2, 2)
Ridgebeta = np.linalg.inv(X.T @ X+l*I) @ (X.T @ y)
print("Own inversion\n")
print(Ridgebeta)

#Gradient Descent
GD = Gradient_Descent_Algorithms(x, y)

print("GD analytical expression to find gradient\n")
#OLS
print("Using OLS cost function\n")
y_pred, beta = GD.GradientDescent("OLS")
y_pred_m, beta_m = GD.GradientDescent("OLS", momentum=True)
GD.Compare_plot("OLS", y_pred, y_pred_m)
print(f"Plain gradient descent beta: {beta}\n")
print(f"Gradient descent with momentum beta: {beta_m}\n")

#Ridge
print("Using Ridge cost function\n")
y_pred, beta = GD.GradientDescent("Ridge", l)
y_pred_m, beta_m = GD.GradientDescent("Ridge", l, momentum=True)
GD.Compare_plot("Ridge", y_pred, y_pred_m)
print(f"Plain gradient descent beta: {beta}\n")
print(f"Gradient descent with momentum beta: {beta_m}\n")

print("GD AutoGrad to find gradient\n")
#OLS
print("Using OLS cost function\n")
y_pred, beta = GD.GradientDescent_AutoGrad("OLS")
y_pred_m, beta_m = GD.GradientDescent_AutoGrad("OLS", momentum=True)
GD.Compare_plot("OLS", y_pred, y_pred_m)
print(f"Plain gradient descent beta: {beta}\n")
print(f"Gradient descent with momentum beta: {beta_m}\n")

#Ridge
print("Using Ridge cost function\n")
y_pred, beta = GD.GradientDescent_AutoGrad("Ridge", l)
y_pred_m, beta_m = GD.GradientDescent_AutoGrad("Ridge", l, momentum=True)
GD.Compare_plot("Ridge", y_pred, y_pred_m)
print(f"Plain gradient descent beta: {beta}\n")
print(f"Gradient descent with momentum beta: {beta_m}\n")

#Stochastic Gradient Descent
SGD = Stochastic_GD_Algorithms(x, y)

n_epochs = 50
batch_size = 5

print("SGD analytical expression to find gradient\n")
#OLS
print("Using OLS cost function\n")
y_pred, beta = SGD.Stochastic_gradient_descent(n_epochs, batch_size, "OLS")
y_pred_m, beta_m = SGD.Stochastic_gradient_descent(n_epochs, batch_size, "OLS", momentum=True)
SGD.Compare_plot("OLS", y_pred, y_pred_m)
print(f"Stochastic gradient descent beta: {beta}\n")
print(f"Stochastic gradient descent with momentum beta: {beta_m}\n")

#Ridge
print("Using Ridge cost function\n")
y_pred, beta  = SGD.Stochastic_gradient_descent(n_epochs, batch_size, "Ridge", l)
y_pred_m, beta_m = SGD.Stochastic_gradient_descent(n_epochs, batch_size, "Ridge", l, momentum=True)
SGD.Compare_plot("Ridge", y_pred, y_pred_m)
print(f"Stochastic gradient descent beta: {beta}\n")
print(f"Stochastic gradient descent with momentum beta: {beta_m}\n")

print("SGD AutoGrad to find gradient")
#OLS
print("Using OLS cost function\n")
y_pred, beta = SGD.SGD_AutoGrad(n_epochs, batch_size, "OLS")
y_pred_m, beta_m = SGD.SGD_AutoGrad(n_epochs, batch_size, "OLS", momentum=True)
SGD.Compare_plot("OLS", y_pred, y_pred_m)
print(f"Stochastic gradient descent beta: {beta}\n")
print(f"Stochastic gradient descent with momentum beta: {beta_m}\n")

#Ridge
print("Using Ridge cost function\n")
y_pred, beta  = SGD.SGD_AutoGrad(n_epochs, batch_size, "Ridge", l)
y_pred_m, beta_m = SGD.SGD_AutoGrad(n_epochs, batch_size, "Ridge", l, momentum=True)
SGD.Compare_plot("Ridge", y_pred, y_pred_m)
print(f"Stochastic gradient descent beta: {beta}\n")
print(f"Stochastic gradient descent with momentum beta: {beta_m}\n")

print("Different methods for tuning the learning rate with analytical expression for the gradient\n")
#AdaGrad with and without momentum for plain gradient descent and SGD.
#First for Cost function defined by Ordinary least squares
print("Using OLS cost function\n")
TLR = TuningLearningRate(x,y)
#plain gradient descent without momentum
y_pred_OLS_GD_AdaGrad, beta = TLR.tuning_learning_rate("OLS", "GD", "AdaGrad")
#plain gradient descent with momentum
y_pred_OLS_GD_AdaGrad_m, beta_m = TLR.tuning_learning_rate("OLS", "GD", "AdaGrad", momentum= True)

#Plot
TLR.Compare_plot("AdaGrad", "GD", "OLS", y_pred_OLS_GD_AdaGrad, y_pred_OLS_GD_AdaGrad_m)
print(f"Plain gradient descent beta: {beta} using AdaGrad\n")
print(f"Gradient descent with momentum beta: {beta_m} using AdaGrad\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_AdaGrad, beta = TLR.tuning_learning_rate("OLS", "SGD", "AdaGrad")
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_AdaGrad_m, beta_m = TLR.tuning_learning_rate("OLS", "SGD", "AdaGrad", momentum=True)

#Plot
TLR.Compare_plot("AdaGrad", "SGD", "OLS", y_pred_OLS_SGD_AdaGrad, y_pred_OLS_SGD_AdaGrad_m)
print(f"Stochastic gradient descent beta: {beta} using AdaGrad\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using AdaGrad\n")

#AdaGrad with and without momentum for plain gradient descent and SGD.
#Second, for Cost function defined by Ridge Regression
print("Using Ridge cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_AdaGrad, beta = TLR.tuning_learning_rate("Ridge", "GD", "AdaGrad\n", l)
#plain gradient descent with momentum
y_pred_OLS_GD_AdaGrad_m, beta_m = TLR.tuning_learning_rate("Ridge", "GD", "AdaGrad\n", l, momentum= True)

#Plot
TLR.Compare_plot("AdaGrad", "GD", "Ridge", y_pred_OLS_GD_AdaGrad, y_pred_OLS_GD_AdaGrad_m)
print(f"Plain gradient descent beta: {beta} using AdaGrad\n")
print(f"Gradient descent with momentum beta: {beta_m} using AdaGrad\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_AdaGrad, beta = TLR.tuning_learning_rate("Ridge", "SGD", "AdaGrad", l)
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_AdaGrad_m, beta_m = TLR.tuning_learning_rate("Ridge", "SGD", "AdaGrad", l, momentum=True)

#Plot
TLR.Compare_plot("AdaGrad", "SGD", "Ridge", y_pred_OLS_SGD_AdaGrad, y_pred_OLS_SGD_AdaGrad_m)
print(f"Stochastic gradient descent beta: {beta} using AdaGrad\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using AdaGrad\n")

#RMSProp with and without momentum for plain gradient descent and SGD.
#First for Cost function defined by Ordinary least squares
print("Using OLS cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_RMSProp, beta = TLR.tuning_learning_rate("OLS", "GD", "RMSProp")
#plain gradient descent with momentum
y_pred_OLS_GD_RMSProp_m, beta_m = TLR.tuning_learning_rate("OLS", "GD", "RMSProp", momentum= True)

#Plot
TLR.Compare_plot("RMSProp", "GD", "OLS", y_pred_OLS_GD_RMSProp, y_pred_OLS_GD_RMSProp_m)
print(f"Plain gradient descent beta: {beta} using RMSProp\n")
print(f"Gradient descent with momentum beta: {beta_m} using RMSProp\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_RMSProp, beta = TLR.tuning_learning_rate("OLS", "SGD", "RMSProp")
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_RMSProp_m, beta_m = TLR.tuning_learning_rate("OLS", "SGD", "RMSProp", momentum=True)

#Plot
TLR.Compare_plot("RMSProp", "SGD", "OLS", y_pred_OLS_SGD_RMSProp, y_pred_OLS_SGD_RMSProp_m)
print(f"Stochastic gradient descent beta: {beta} using RMSProp\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using RMSProp\n")

#RMSProp with and without momentum for plain gradient descent and SGD.
#Second for Cost function defined by Ridge Regression
print("Using Ridge cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_RMSProp, beta = TLR.tuning_learning_rate("Ridge", "GD", "RMSProp", l)
#plain gradient descent with momentum
y_pred_OLS_GD_RMSProp_m, beta_m = TLR.tuning_learning_rate("Ridge", "GD", "RMSProp", l, momentum= True)

#Plot
TLR.Compare_plot("RMSProp", "GD", "Ridge", y_pred_OLS_GD_RMSProp, y_pred_OLS_GD_RMSProp_m)
print(f"Plain gradient descent beta: {beta} using RMSProp\n")
print(f"Gradient descent with momentum beta: {beta_m} using RMSProp\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_RMSProp, beta = TLR.tuning_learning_rate("Ridge", "SGD", "RMSProp", l)
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_RMSProp_m, beta_m = TLR.tuning_learning_rate("Ridge", "SGD", "RMSProp", l, momentum=True)

#Plot
TLR.Compare_plot("RMSProp", "SGD", "Ridge", y_pred_OLS_SGD_RMSProp, y_pred_OLS_SGD_RMSProp_m)
print(f"Stochastic gradient descent beta: {beta} using RMSProp\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using RMSProp\n")

#ADAM with and without momentum for plain gradient descent and SGD.
#First for Cost function defined by Ordinary least squares
print("Using OLS cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_ADAM, beta = TLR.tuning_learning_rate("OLS", "GD", "ADAM")
#plain gradient descent with momentum
y_pred_OLS_GD_ADAM_m, beta_m = TLR.tuning_learning_rate("OLS", "GD", "ADAM", momentum= True)

#Plot
TLR.Compare_plot("ADAM", "GD", "OLS", y_pred_OLS_GD_ADAM, y_pred_OLS_GD_ADAM_m)
print(f"Plain gradient descent beta: {beta} using ADAM\n")
print(f"Gradient descent with momentum beta: {beta_m} using ADAM\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_ADAM, beta = TLR.tuning_learning_rate("OLS", "SGD", "ADAM")
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_ADAM_m, beta_m = TLR.tuning_learning_rate("OLS", "SGD", "ADAM", momentum=True)

#Plot
TLR.Compare_plot("ADAM", "SGD", "OLS", y_pred_OLS_SGD_ADAM, y_pred_OLS_SGD_ADAM_m)
print(f"Stochastic gradient descent beta: {beta} using ADAM\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using ADAM\n")

#ADAM with and without momentum for plain gradient descent and SGD.
#Second for Cost function defined by Ridge Regression
print("Using Ridge cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_ADAM, beta = TLR.tuning_learning_rate("Ridge", "GD", "ADAM", l)
#plain gradient descent with momentum
y_pred_OLS_GD_ADAM_m, beta_m = TLR.tuning_learning_rate("Ridge", "GD", "ADAM", l, momentum= True)

#Plot
TLR.Compare_plot("ADAM", "GD", "Ridge", y_pred_OLS_GD_ADAM, y_pred_OLS_GD_ADAM_m)
print(f"Plain gradient descent beta: {beta} using ADAM\n")
print(f"Gradient descent with momentum beta: {beta_m} using ADAM\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_ADAM, beta = TLR.tuning_learning_rate(x, y, "Ridge", "SGD", "ADAM", l)
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_ADAM_m, beta_m = TLR.tuning_learning_rate(x, y, "Ridge", "SGD", "ADAM", l, momentum=True)

#Plot
TLR.Compare_plot("ADAM", "SGD", "Ridge", y_pred_OLS_SGD_ADAM, y_pred_OLS_SGD_ADAM_m)
print(f"Stochastic gradient descent beta: {beta} using ADAM\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using ADAM\n")


print("Different methods for tuning the learning rate with AutoGrad for finding the gradient\n")
#AdaGrad with and without momentum for plain gradient descent and SGD.
#First for Cost function defined by Ordinary least squares
print("Using OLS cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_AdaGrad, beta = TLR.TLR_autograd("OLS", "GD", "AdaGrad")
#plain gradient descent with momentum
y_pred_OLS_GD_AdaGrad_m, beta_m = TLR.TLR_autograd("OLS", "GD", "AdaGrad", momentum= True)

#Plot
TLR.Compare_plot("AdaGrad", "GD", "OLS", y_pred_OLS_GD_AdaGrad, y_pred_OLS_GD_AdaGrad_m)
print(f"Plain gradient descent beta: {beta} using AdaGrad\n")
print(f"Gradient descent with momentum beta: {beta_m} using AdaGrad\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_AdaGrad, beta = TLR.TLR_autograd("OLS", "SGD", "AdaGrad")
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_AdaGrad_m, beta_m = TLR.TLR_autograd("OLS", "SGD", "AdaGrad", momentum=True)

#Plot
TLR.Compare_plot("AdaGrad", "SGD", "OLS", y_pred_OLS_SGD_AdaGrad, y_pred_OLS_SGD_AdaGrad_m)
print(f"Stochastic gradient descent beta: {beta} using AdaGrad\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using AdaGrad\n")

#AdaGrad with and without momentum for plain gradient descent and SGD.
#Second, for Cost function defined by Ridge Regression
print("Using Ridge cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_AdaGrad, beta = TLR.TLR_autograd("Ridge", "GD", "AdaGrad", l)
#plain gradient descent with momentum
y_pred_OLS_GD_AdaGrad_m, beta_m = TLR.TLR_autograd("Ridge", "GD", "AdaGrad", l, momentum= True)

#Plot
TLR.Compare_plot("AdaGrad", "GD", "Ridge", y_pred_OLS_GD_AdaGrad, y_pred_OLS_GD_AdaGrad_m)
print(f"Plain gradient descent beta: {beta} using AdaGrad\n")
print(f"Gradient descent with momentum beta: {beta_m} using AdaGrad\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_AdaGrad, beta = TLR.TLR_autograd("Ridge", "SGD", "AdaGrad", l)
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_AdaGrad_m, beta_m = TLR.TLR_autograd("Ridge", "SGD", "AdaGrad", l, momentum=True)

#Plot
TLR.Compare_plot("AdaGrad", "SGD", "Ridge", y_pred_OLS_SGD_AdaGrad, y_pred_OLS_SGD_AdaGrad_m)
print(f"Stochastic gradient descent beta: {beta} using AdaGrad\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using AdaGrad\n")

#RMSProp with and without momentum for plain gradient descent and SGD.
#First for Cost function defined by Ordinary least squares
print("Using OLS cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_RMSProp, beta = TLR.TLR_autograd("OLS", "GD", "RMSProp")
#plain gradient descent with momentum
y_pred_OLS_GD_RMSProp_m, beta_m = TLR.TLR_autograd("OLS", "GD", "RMSProp", momentum= True)

#Plot
TLR.Compare_plot("RMSProp", "GD", "OLS", y_pred_OLS_GD_RMSProp, y_pred_OLS_GD_RMSProp_m)
print(f"Plain gradient descent beta: {beta} using RMSProp\n")
print(f"Gradient descent with momentum beta: {beta_m} using RMSProp\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_RMSProp, beta = TLR.TLR_autograd("OLS", "SGD", "RMSProp")
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_RMSProp_m, beta_m = TLR.TLR_autograd("OLS", "SGD", "RMSProp", momentum=True)

#Plot
TLR.Compare_plot("RMSProp", "SGD", "OLS", y_pred_OLS_SGD_RMSProp, y_pred_OLS_SGD_RMSProp_m)
print(f"Stochastic gradient descent beta: {beta} using RMSProp\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using RMSProp\n")

#RMSProp with and without momentum for plain gradient descent and SGD.
#Second for Cost function defined by Ridge Regression
print("Using Ridge cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_RMSProp, beta = TLR.TLR_autograd("Ridge", "GD", "RMSProp", l)
#plain gradient descent with momentum
y_pred_OLS_GD_RMSProp_m, beta_m = TLR.TLR_autograd("Ridge", "GD", "RMSProp", l, momentum= True)

#Plot
TLR.Compare_plot("RMSProp", "GD", "Ridge", y_pred_OLS_GD_RMSProp, y_pred_OLS_GD_RMSProp_m)
print(f"Plain gradient descent beta: {beta} using RMSProp\n")
print(f"Gradient descent with momentum beta: {beta_m} using RMSProp\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_RMSProp, beta = TLR.TLR_autograd("Ridge", "SGD", "RMSProp", l)
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_RMSProp_m, beta_m = TLR.TLR_autograd(x, y, "Ridge", "SGD", "RMSProp", l, momentum=True)

#Plot
TLR.Compare_plot("RMSProp", "SGD", "Ridge", y_pred_OLS_SGD_RMSProp, y_pred_OLS_SGD_RMSProp_m)
print(f"Stochastic gradient descent beta: {beta} using RMSProp\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using RMSProp\n")

#ADAM with and without momentum for plain gradient descent and SGD.
#First for Cost function defined by Ordinary least squares
print("Using OLS cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_ADAM, beta = TLR.TLR_autograd("OLS", "GD", "ADAM")
#plain gradient descent with momentum
y_pred_OLS_GD_ADAM_m, beta_m = TLR.TLR_autograd("OLS", "GD", "ADAM", momentum= True)

#Plot
TLR.Compare_plot("ADAM", "GD", "OLS", y_pred_OLS_GD_ADAM, y_pred_OLS_GD_ADAM_m)
print(f"Plain gradient descent beta: {beta} using ADAM\n")
print(f"Gradient descent with momentum beta: {beta_m} using ADAM\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_ADAM, beta = TLR.TLR_autograd("OLS", "SGD", "ADAM")
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_ADAM_m, beta_m = TLR.TLR_autograd("OLS", "SGD", "ADAM", momentum=True)

#Plot
TLR.Compare_plot("ADAM", "SGD", "OLS", y_pred_OLS_SGD_ADAM, y_pred_OLS_SGD_ADAM_m)
print(f"Stochastic gradient descent beta: {beta} using ADAM\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using ADAM\n")

#ADAM with and without momentum for plain gradient descent and SGD.
#Second for Cost function defined by Ridge Regression
print("Using Ridge cost function\n")
#plain gradient descent without momentum
y_pred_OLS_GD_ADAM, beta = TLR.TLR_autograd("Ridge", "GD", "ADAM", l)
#plain gradient descent with momentum
y_pred_OLS_GD_ADAM_m, beta_m = TLR.TLR_autograd("Ridge", "GD", "ADAM", l, momentum= True)

#Plot
TLR.Compare_plot("ADAM", "GD", "Ridge", y_pred_OLS_GD_ADAM, y_pred_OLS_GD_ADAM_m)
print(f"Plain gradient descent beta: {beta} using ADAM\n")
print(f"Gradient descent with momentum beta: {beta_m} using ADAM\n")

#Stochastic gradient descent without momentum
y_pred_OLS_SGD_ADAM, beta = TLR.TLR_autograd("Ridge", "SGD", "ADAM", l)
#Stochastic gradient descent with momentum
y_pred_OLS_SGD_ADAM_m, beta_m = TLR.TLR_autograd("Ridge", "SGD", "ADAM", l, momentum=True)

#Plot
TLR.Compare_plot("ADAM", "SGD", "Ridge", y_pred_OLS_SGD_ADAM, y_pred_OLS_SGD_ADAM_m)
print(f"Stochastic gradient descent beta: {beta} using ADAM\n")
print(f"Stochastic gradient descentt with momentum beta: {beta_m} using ADAM\n")

#Neural Network
#Function for regression task


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Design matrix:
def designMatrix(x,y, degree):
    n = degree
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) #number of elements in beta
    X = np.ones((N,l))

    #Adding elements in the design matrix X on the from [x, y, x**2, y**2, xy, ...]
    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

target = FrankeFunction(x, y)
target = target.reshape(target.shape[0], 1)
X = designMatrix(x, y, degree = 3)
X_train, X_test, y_train, y_test = train_test_split(X, target)

# visual representation of grid search
# uses seaborn heatmap, you can also do this with matplotlib imshow
import seaborn as sns
def vizualization(nn_regressor):
    #learning parameter
    eta_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.logspace(-5, 0, 6)
    sns.set()

    test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_r2score = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            optim = Constant(eta=eta_vals[i])
            nn_regressor.reset_weights() # reset weights such that previous runs or reruns don't affect the weights
            nn_regressor.train(X_train, y_train, optim, lam = lmbd_vals[j])

            # Test the trained model on the test set
            test_predictions = nn_regressor.predict(X_test)
            test_mse[i][j] = np.mean((y_test - test_predictions) ** 2)
            test_r2score[i][j] = nn_regressor.R2(test_predictions, y_test)

            
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_r2score, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test R2 Score")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

#Regularzation terms: no hidden layer and one output node.
input_nodes = X_train.shape[1]
output_nodes = 1
#First using the Neural network for regression, using the OLS cost function
linear_regression = Neural_Network((input_nodes, output_nodes), output_func=sigmoid, cost_func=CostOLS, seed=2023)
vizualization(linear_regression)

#Second using the Neural network for regression, using the Ridge cost function
linear_regression = Neural_Network((input_nodes, output_nodes), output_func=sigmoid, cost_func=RidgeCost, seed=2023)
vizualization(linear_regression)

#Regularzation terms: one hidden layer and one output node.
hidden_nodes = 2
#First using the Neural network for regression, using the OLS cost function
linear_regression = Neural_Network((input_nodes, hidden_nodes, output_nodes), output_func=sigmoid, cost_func=CostOLS, seed=2023)
vizualization(linear_regression)

#Second using the Neural network for regression, using the Ridge cost function
linear_regression = Neural_Network((input_nodes, hidden_nodes, output_nodes), output_func=sigmoid, cost_func=RidgeCost, seed=2023)
vizualization(linear_regression)

#activation function: RELU
#cost function: OLS
linear_regression = Neural_Network((input_nodes, hidden_nodes, output_nodes), activation_func=RELU ,output_func=sigmoid, cost_func=CostOLS, seed=2023)
vizualization(linear_regression)

#activation function: RELU
#cost function: Ridge
linear_regression = Neural_Network((input_nodes, hidden_nodes, output_nodes), activation_func=RELU ,output_func=sigmoid, cost_func=RidgeCost, seed=2023)
vizualization(linear_regression)

#activation function: LRELU
#cost function: OLS
linear_regression = Neural_Network((input_nodes, hidden_nodes, output_nodes), activation_func=LRELU ,output_func=sigmoid, cost_func=CostOLS, seed=2023)
vizualization(linear_regression)

#activation function: LRELU
#cost function: Ridge
linear_regression = Neural_Network((input_nodes, hidden_nodes, output_nodes), activation_func=LRELU ,output_func=sigmoid, cost_func=RidgeCost, seed=2023)
vizualization(linear_regression)

#Classification
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()      #Download breast cancer dataset

inputs=cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
outputs=cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
labels=cancer.feature_names[0:30]

print('The content of the breast cancer dataset is:')      #Print information about the datasets
print(labels)
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))

x=inputs      #Reassign the Feature and Label matrices to other variables
y=outputs

y = y.reshape(y.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(x, y)
#Regularzation terms: no hidden layer and one output node.
input_nodes = X_train.shape[1]
output_nodes = 1
hidden_nodes = 2
#Classification task
classification = Neural_Network((input_nodes, hidden_nodes, output_nodes), output_func=sigmoid, cost_func=CrossEntropy, seed=2023)

optim = Constant(eta=1e-4)
classification.reset_weights() # reset weights such that previous runs or reruns don't affect the weights
scores = classification.train(X_train, y_train, optim, lam = 0.01,X_val= X_test, y_target=y_test)
pred = classification.predict(X_test)
acc = classification.accuracy_score(pred, y_test)
print(acc)


#Logistic Regression
def plot(test, train, lam):
    plt.plot(lam, test, label = "test accuracy")
    plt.plot(lam, train, label = "train accuracy")
    plt.legend()
    plt.title("Accuracy score Logistic regression")
    plt.show()

# Instantiate and train the logistic regression model
lam_vals = np.logspace(-5, 0, 6)
test = list()
train = list()

for lam in lam_vals:
    logreg_sgd = LogisticRegressionSGD(learning_rate=lam, max_iters=1000, tol=1e-4)
    logreg_sgd.fit(X_train, y_train)
    train_accuracy = logreg_sgd.accuracy(X_train, y_train)
    test_accuracy = logreg_sgd.accuracy(X_test, y_test)
    train.append(train_accuracy)
    test.append(test_accuracy)

plot(test, train, lam_vals)