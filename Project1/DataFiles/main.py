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
from imageio import imread
from numpy.random import normal, uniform
import Function
import Plots

#Franke function
np.random.seed(0)
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


#Adding noise to the Franke function:
z += np.random.normal(0, 1)
z = np.ravel(z)

#Plotting MSE and R2 score OLS
polydegree, TestError, TrainError, Test_R2, Train_R2 = OLS(x, y, z, 5,"Franke", True)
plot_stat(TestError, TrainError, "MSE", polydegree)
plot_stat(Test_R2, Train_R2, "R2_Score", polydegree)


#Ridge regression plot with MSE as funtion of lambda
for degree in range(1, 6):
    lambdas, MSERidgePredict, MSERidgeTrain, R2RidgePredict, R2RidgeTrain = RidgeOrLasso(x, y, z, degree, "Ridge")
    print(f"MSE Train for OLS with polynomial fit of degree {degree}: {TrainError[degree-1]}")
    print(f"MSE Test for OLS with polynomial fit of degree {degree}: {TestError[degree-1]}")
    plt.figure()
    plt.plot(np.log10(lambdas), MSERidgeTrain, label = 'MSE Ridge Train')
    plt.plot(np.log10(lambdas), MSERidgePredict, 'r--', label = 'MSE Ridge Test')
    plt.xlabel('log10(lambda)')
    plt.ylabel('MSE')
    plt.title(f"MSE Ridge Regression for a polynomial fit of degree {degree}")
    plt.legend()
    plt.show()

    print(f"R2 Score Train for OLS with polynomial fit of degree {degree}: {Train_R2[degree-1]}")
    print(f"R2 Score Test for OLS with polynomial fit of degree {degree}: {Test_R2[degree-1]}")
    plt.figure()
    plt.plot(np.log10(lambdas), R2RidgeTrain, label = 'R2 Score Ridge Train')
    plt.plot(np.log10(lambdas), R2RidgePredict, 'r--', label = 'R2 Score Ridge Test')
    plt.xlabel('log10(lambda)')
    plt.ylabel('R2 Score')
    plt.title(f"R2 Score Ridge Regression for a polynomial fit of degree {degree}")
    plt.legend()
    plt.show()

#Plots of Ridge VS Lasso where MSE is function of lambda
for degree in range(1, 6):
    lambdas, MSERidgePredict, MSERidgeTrain, R2RidgePredict, R2RidgeTrain = RidgeOrLasso(x, y, z, degree, "Ridge")
    lambdas, MSELassoPredict, MSELassoTrain, R2LassoPredict, R2LassoTrain = RidgeOrLasso(x, y, z, degree, "Lasso")
    plot_stat_RvsL(degree, lambdas, MSERidgeTrain, MSERidgePredict, MSELassoPredict, MSELassoTrain, "MSE")
    plot_stat_RvsL(degree, lambdas, R2RidgePredict, R2RidgeTrain, R2LassoPredict, R2LassoTrain, "R2")


#Bias-Varianse Tradeoff + bootstrap
polydegree, TestError, TrainError, Test_R2, Train_R2 = OLS(x, y, z, 5, plot= False)
plt.plot(polydegree, TestError, label='Test')
plt.title("MSE of test data as a function of polynomial degree")
plt.xlabel("Complexity")
plt.ylabel("Prediction Error")
plt.legend()
plt.show()

maxdegree = 5

bias_variance_tradeoff(x, y, z, maxdegree, 10)
bias_variance_tradeoff(x, y, z, maxdegree, 50)
bias_variance_tradeoff(x ,y, z, maxdegree, 100)

#Cross validations
Cross_Validation_plot(x, y, z)


#Terrain Data
# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')

#Since the shape of the original array is (3601, 1801) we will just look at a part of the terrain data, say  with shape (1000, 1000)
N = 1000
m = 5 # polynomial order
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

z_terrain = terrain

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#Plotting MSE and R2 score as function of polynomial degree
z_terrain = z_terrain.ravel()
polydegree, TestError, TrainError, Test_R2, Train_R2 = OLS(x_mesh, y_mesh, z_terrain, m, "Terrain", False)
plot_stat(TestError, TrainError, "MSE", polydegree)
plot_stat(Test_R2, Train_R2, "R2_Score", polydegree)

#Plotting MSE and R2 score as funtion of lambda for Ridge regression
for degree in range(1, 6):
    lambdas, MSERidgePredict, MSERidgeTrain, R2RidgePredict, R2RidgeTrain = RidgeOrLasso(x_mesh, y_mesh, z_terrain, degree, "Ridge")
    print(f"MSE Train for OLS with polynomial fit of degree {degree}: {TrainError[degree-1]}")
    print(f"MSE Test for OLS with polynomial fit of degree {degree}: {TestError[degree-1]}")
    plt.figure()
    plt.plot(np.log10(lambdas), MSERidgeTrain, label = 'MSE Ridge Train')
    plt.plot(np.log10(lambdas), MSERidgePredict, 'r--', label = 'MSE Ridge Test')
    plt.xlabel('log10(lambda)')
    plt.ylabel('MSE')
    plt.title(f"MSE Ridge Regression for a polynomial fit of degree {degree}")
    plt.legend()
    plt.show()

    print(f"R2 Score Train for OLS with polynomial fit of degree {degree}: {Train_R2[degree-1]}")
    print(f"R2 Score Test for OLS with polynomial fit of degree {degree}: {Test_R2[degree-1]}")
    plt.figure()
    plt.plot(np.log10(lambdas), R2RidgeTrain, label = 'R2 Score Ridge Train')
    plt.plot(np.log10(lambdas), R2RidgePredict, 'r--', label = 'R2 Score Ridge Test')
    plt.xlabel('log10(lambda)')
    plt.ylabel('R2 Score')
    plt.title(f"R2 Score Ridge Regression for a polynomial fit of degree {degree}")
    plt.legend()
    plt.show()

#Ridge VS Lasso
for degree in range(1, 6):
    lambdas, MSERidgePredict, MSERidgeTrain, R2RidgePredict, R2RidgeTrain = RidgeOrLasso(x_mesh, y_mesh, z_terrain, degree, "Ridge")
    lambdas, MSELassoPredict, MSELassoTrain, R2LassoPredict, R2LassoTrain = RidgeOrLasso(x_mesh, y_mesh, z_terrain, degree, "Lasso")
    plot_stat_RvsL(degree, lambdas, MSERidgeTrain, MSERidgePredict, MSELassoPredict, MSELassoTrain, "MSE")
    plot_stat_RvsL(degree, lambdas, R2RidgePredict, R2RidgeTrain, R2LassoPredict, R2LassoTrain, "R2")


polydegree, TestError, TrainError, Test_R2, Train_R2 = OLS(x_mesh, y_mesh, z_terrain, 5, plot= False)
plt.plot(polydegree, TestError, label='Test')
plt.title("MSE of test data as a function of polynomial degree")
plt.xlabel("Complexity")
plt.ylabel("Prediction Error")
plt.legend()
plt.show()

#Bias-Variance tradeoff + bootstrap
bias_variance_tradeoff(x_mesh, y_mesh, z_terrain, m, 10)
bias_variance_tradeoff(x_mesh, y_mesh, z_terrain, m, 50)
bias_variance_tradeoff(x_mesh ,y_mesh, z_terrain, m, 100)


#Cross validation
#Since the shape of the original array is (3601, 1801) we will just look at a part of the terrain data, say  with shape (1000, 1000)
N = 100
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

z_terrain = terrain
z_terrain = z_terrain.ravel()

Cross_Validation_plot(x_mesh, y_mesh, z_terrain)