#All packages need for this assignment:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#All the functions needed:

class Functions:
    def MSE(y_data,y_model):
        n = np.size(y_model)
        return np.sum((y_data-y_model)**2)/n

    def R2(y_data, y_model):
        return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

    # SVD inversion
    def SVDinv(A):
        U, s, VT = np.linalg.svd(A)
        # reciprocals of singular values of s
        d = 1.0 / s
        # create m x n D matrix
        D = np.zeros(A.shape)
        # populate D with n x n diagonal matrix
        D[:A.shape[1], :A.shape[1]] = np.diag(d)
        UT = np.transpose(U)
        V = np.transpose(VT)
        return np.matmul(V,np.matmul(D.T,UT))
    
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
    
    def OLS(x, y, z, maxdegree, data = "Franke", plot = False):
        TestError = np.zeros(maxdegree)
        TrainError = np.zeros(maxdegree)
        Test_R2 = np.zeros(maxdegree)
        Train_R2 = np.zeros(maxdegree)
        polydegree = np.zeros(maxdegree)
        for degree in range(1,maxdegree+1):
            X = designMatrix(x, y, degree)
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
            #Ordinary least squares: for given degree
            # matrix inversion to find beta
            beta = (SVDinv(X_train.T@X_train)@X_train.T)@z_train
            #prediction
            z_fit = X_train@beta
            z_pred = X_test@beta
            if data == "Franke":
                if plot == True:
                    plot_beta(X, z, beta, degree)
            elif data == "Terrain":
                if plot == True:
                    plot_terrain(X, beta, degree)
            #Adding elements to lists for plotting
            index = degree-1
            polydegree[index] = degree
            TestError[index] = MSE(z_test, z_pred)
            TrainError[index] = MSE(z_train, z_fit)
            Test_R2[index] = R2(z_test, z_pred)
            Train_R2[index] = R2(z_train, z_fit)
            
        return polydegree, TestError, TrainError, Test_R2, Train_R2
    
    def RidgeOrLasso(x, y, z, degree, type):
        nlambdas = 100
        lambdas = np.logspace(-4, 4, nlambdas)
        MSEPredict = np.zeros(nlambdas)
        MSETrain = np.zeros(nlambdas)
        R2Predict = np.zeros(nlambdas)
        R2Train = np.zeros(nlambdas)
        betas = []

        X = designMatrix(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        for i in range(nlambdas):
            lmb = lambdas[i]
            if type == "Ridge":
                I = np.identity(np.size(X_train, 1))
                beta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ z_train
                # make the prediction
                z_tilde = X_train @ beta
                z_predict = X_test @ beta
            if type == "Lasso":
                RegLasso = linear_model.Lasso(lmb, fit_intercept = False)
                RegLasso.fit(X_train, z_train)
                z_tilde = RegLasso.predict(X_train)
                z_predict= RegLasso.predict(X_test)
                beta = RegLasso.coef_
                
            MSEPredict[i] = MSE(z_test, z_predict)
            MSETrain[i] = MSE(z_train, z_tilde)
            R2Predict[i] = R2(z_test, z_predict)
            R2Train[i] = R2(z_train, z_tilde)
            betas.append(beta)
            

        return lambdas, MSEPredict, MSETrain, R2Predict, R2Train
    
    def bias_variance_tradeoff(x, y, z, maxdegree, n_boostraps):
        error = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)
        polydegree = np.zeros(maxdegree)

        for degree in range(1, maxdegree+1):
            X = designMatrix(x,y, degree)
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
            z_pred = np.empty((z_test.shape[0], n_boostraps))
            for i in range(n_boostraps):
                x_, z_ = resample(X_train, z_train)
                beta = (SVDinv(x_.T@x_)@x_.T)@z_
                z_pred[:, i] = X_test@beta

            index = degree-1
            polydegree[index] = degree
            # Calculate error, bias, and variance
            error[index] = np.mean(np.mean((z_test[:, np.newaxis] - z_pred) ** 2, axis=1))
            bias[index] = np.mean((z_test - np.mean(z_pred, axis=1)) ** 2)
            variance[index] = np.mean(np.var(z_pred, axis=1))
            print('Polynomial degree:', degree)
            print('Error:', error[index])
            print('Bias^2:', bias[index])
            print('Var:', variance[index])
            print('{} >= {} + {} = {}'.format(error[index], bias[index], variance[index], bias[index]+variance[index]))

        plt.plot(polydegree, error, label='Error')
        plt.plot(polydegree, bias, label='bias')
        plt.plot(polydegree, variance, label='Variance')
        plt.xlabel('Polynomial Degree')
        plt.title("Bias-Variance tradeoff")
        plt.legend()
        plt.show()

    def k_fold_cross_validation(x, y, z, degree, type, k=5, lmb=1):
        X = designMatrix(x, y, degree)
        n = len(X)
        fold_size = n // k
        scores_KFold = []

        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size

            X_test = X[start:end]
            z_test = z[start:end]

            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            z_train = np.concatenate((z[:start], z[end:]), axis=0)

            if type == "OLS":
                beta = (SVDinv(X_train.T@X_train)@X_train.T)@z_train
                z_tilde = X_train@beta
                z_pred = X_test@beta
            elif type == "Ridge":
                I = np.identity(np.size(X_train, 1))
                beta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ z_train
                z_tilde = X_train@beta
                z_pred = X_test@beta
            elif type == "Lasso":
                RegLasso = linear_model.Lasso(lmb, fit_intercept = False)
                RegLasso.fit(X_train, z_train)
                z_tilde = RegLasso.predict(X_train)
                z_pred= RegLasso.predict(X_test)
                beta = RegLasso.coef_


            scores_KFold.append(np.sum((z_pred - z_test[:, np.newaxis])**2)/np.size(z_pred))
        
        estimated_mse_KFold = np.mean(scores_KFold)
        return estimated_mse_KFold
    
