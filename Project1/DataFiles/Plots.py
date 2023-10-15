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

class Plots:
    #Plotting MSE or R^2 score
    def plot_stat(stat_test, stat_train, stat_text, polydegree):
        plt.plot(polydegree, stat_test, 'r--', label='Test')
        plt.plot(polydegree, stat_train, label='Train')
        if stat_text == "MSE":
            plt.title("Mean Squared Error as a function of polynomial degree")
            plt.ylabel(stat_text)
        elif stat_text == "R2_Score":
            plt.title("R2 Score as a function of polynomial degree")
            plt.ylabel(stat_text)
        plt.xlabel("degree")

        plt.legend()
        plt.show()

    #Plotting the betas in a 3D-plot
    def plot_beta(X, z, beta, degree):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #x, y = np.meshgrid(x,y)
        # Plot the surface.
        # Plot the surface.
        surf_1 = ax.scatter3D(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        z_predict = np.reshape(X@beta, x.shape)
        surf_2 = ax.plot_surface(x, y, z_predict, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf_2)
        plt.title(f"Betas for polynomial of degree: {degree}")
        # Make legend, set axes limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()

    def plot_terrain(X, beta, degree):
        # Show the terrain
        plt.figure()
        plt.title(f"Fitted terrain from OLS of polynomial degree {degree}")
        z = np.reshape(X@beta, (1800, 1800))
        plt.imshow(z, cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def plot_stat_RvsL(degree, lambdas, statRidgeTrain, statRidgePredict, statLassoTrain, statLassoPredict, stat):
        plt.figure()
        if stat == "MSE":
            plt.plot(np.log10(lambdas), statRidgeTrain, label = 'MSE Ridge Train')
            plt.plot(np.log10(lambdas), statRidgePredict, 'r--', label = 'MSE Ridge Test')
            plt.plot(np.log10(lambdas), statLassoTrain, label = 'MSE Lasso Train')
            plt.plot(np.log10(lambdas), statLassoPredict, 'g--', label = 'MSE Lasso Test')
            plt.ylabel('MSE')
        elif stat == "R2":
            plt.plot(np.log10(lambdas), statRidgeTrain, label = 'R2 Score Ridge Train')
            plt.plot(np.log10(lambdas), statRidgePredict, 'r--', label = 'R2 Score Ridge Test')
            plt.plot(np.log10(lambdas), statLassoTrain, label = 'R2 Score Lasso Train')
            plt.plot(np.log10(lambdas), statLassoPredict, 'g--', label = 'R2 Score Lasso Test')
            plt.ylabel("R2 Score")

        plt.xlabel('log10(lambda)')
        plt.title(f"Ridge VS Lasso for a polynomial fit of degree {degree}")
        plt.legend()
        plt.show()

    def Cross_Validation_plot(x,y, z):
        degrees = []
        scores = []
        print("Results for OLS")
        for i in range(1, 6):
            degrees.append(i)
            test_scores = k_fold_cross_validation(x, y, z, i, "OLS", k=5, lmb=1)
            scores.append(test_scores)
            # print(f"Polynomial degree {i}, test score: {test_scores}")
        plt.plot(degrees, scores)
        plt.ylabel("mse")
        plt.xlabel("log10(lambda)")
        plt.title("K-fold Cross validation OLS")
        plt.show()

        color = ['r--', 'g--', 'b--', 'y--', 'p--']
        print("Results for Ridge")
        for i in range(1, 6):
            scores = []
            for j in lambdas:
                test_scores = k_fold_cross_validation(x, y, z, i, "Ridge", k=5, lmb=1)
                scores.append(test_scores)
                #print(f"Polynomial degree {i}, lambda = {j},  test score: {test_scores}")
            plt.plot(np.log10(lambdas), scores, color[i-1], label = f"Degree {i}")
        plt.ylabel("mse")
        plt.xlabel("log10(lambda)")
        plt.title("K-fold Cross validation Ridge")
        plt.legend()
        plt.show()

        print("Results for Lasso")
        for i in range(1, 6):
            scores = []
            for j in lambdas:
                test_scores = k_fold_cross_validation(x, y, z, i, "Lasso", k=5, lmb=j)
                scores.append(test_scores)
                #print(f"Polynomial degree {i}, lambda = {j},  test score: {test_scores}")
            
            plt.plot(np.log10(lambdas), scores, color[i-1], label = f"Degree {i}")
        plt.ylabel("mse")
        plt.xlabel("log10(lambda)")
        plt.title("K-fold Cross validation Lasso")
        plt.legend()
        plt.show()
