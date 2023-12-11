import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trialinfo import Trialinfo
from feature_extraction import FeatureExtraction
from MLmethods import ML_methods
from ClassificationTree import ClassificationTree
from DesignMatrix import DesignMatrix

#Design matrices based on raw input data
X_2, X_3, y_2, y_3 = DesignMatrix()

#Logistic regression
X_2_model= ML_methods(X_2, y_2)
X_2_model.LogReg()
X_2_model.LogReg_visualization()

X_3_model = ML_methods(X_3, y_3)
X_3_model.LogReg()
X_3_model.LogReg_visualization()

#Neural network
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
n_hidden_neurons = 50

X_2_model.NN(eta_vals, lmbd_vals, n_hidden_neurons)
X_2_model.NN_visualization()

X_3_model.NN(eta_vals, lmbd_vals, n_hidden_neurons)
X_3_model.NN_visualization()

#Classification tree
tree_X_2 = ClassificationTree(X_2, y_2)
tree_X_2.makeTree()
tree_X_2.printScore()

tree_X_3 = ClassificationTree(X_3, y_3)
tree_X_3.makeTree()
tree_X_3.printScore()

#feature matrices
features = ["min", "max", "mean", "sd", "first_Fourier", "second_Fourier", "third_Fourier", "max_Fourier"]
designMAtrix_X_2 = pd.DataFrame()
extract_2 = FeatureExtraction(X_2)
designMAtrix_X_3 = pd.DataFrame()
extract_3 = FeatureExtraction(X_3)

for feature in features:
    z_2 = extract_2.compute(feature)
    designMAtrix_X_2[feature] = z_2
    z_3 = extract_3.compute(feature)
    designMAtrix_X_3[feature] = z_3

#Logistic regression 
X_2_featuremodel = ML_methods(designMAtrix_X_2, y_2)
X_2_featuremodel.LogReg()
X_2_featuremodel.LogReg_visualization()

X_3_featuremodel = ML_methods(designMAtrix_X_3, y_3)
log = X_3_featuremodel.LogReg()
X_3_featuremodel.LogReg_visualization()

#Neural network
X_2_featuremodel.NN(eta_vals, lmbd_vals, n_hidden_neurons)
X_2_featuremodel.NN_visualization()

X_3_featuremodel.NN(eta_vals, lmbd_vals, n_hidden_neurons)
X_3_featuremodel.NN_visualization()

#Classification Tree
tree_feature_X_2 = ClassificationTree(designMAtrix_X_2, y_2)
tree_feature_X_2.makeTree()
tree_feature_X_2.printScore()

tree_feature_X_3 = ClassificationTree(designMAtrix_X_3, y_3)
tree_feature_X_3.makeTree()
tree_feature_X_3.printScore()

