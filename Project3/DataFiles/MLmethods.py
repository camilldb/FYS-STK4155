from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt

class ML_methods:
    def __init__(self, X, y):
        #Design matrix containing the input and the target variable
        self.X = X
        self.y = y
        #smote sampling
        # os = SMOTE(random_state = 0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

        # self.X_train, self.y_train = os.fit_resample(self.X_train, self.y_train)


    
    #Logistic Regression
    def LogReg(self):
        logreg = LogisticRegression(solver='lbfgs')
        logreg.fit(self.X_train, self.y_train)
        self.test_pred = logreg.predict(self.X_test)
        self.train_pred = logreg.predict(self.X_train)
        #test_score = logreg.score(self.X_test, self.y_test)
        test_score = accuracy_score(self.y_test, self.test_pred)
        print("Test set accuracy with Logistic Regression:", test_score)
        #train_score = logreg.score(self.X_train, self.y_train)
        train_score = accuracy_score(self.y_train, self.train_pred)
        print("Train set accuracy with Logistic Regression:", train_score)
    
    def LogReg_visualization(self):
        #Test set
        skplt.metrics.plot_confusion_matrix(self.y_test, self.test_pred, normalize=True)
        plt.title("Confusion matrix test set")
        plt.show()
        #Train set
        skplt.metrics.plot_confusion_matrix(self.y_train, self.train_pred, normalize=True)
        plt.title("Confusion matrix train set")
        plt.show()
    
    #Neural network - multi layer perceptron
    def NN(self, eta_vals, lmbd_vals, n_hidden_neurons, epochs = 100):
        self.n_eta = len(eta_vals)
        self.n_lmbd = len(lmbd_vals)
        self.NN_list = np.zeros((self.n_eta, self.n_lmbd), dtype=object)

        for i, eta in enumerate(eta_vals):
            for j, lmbd in enumerate(lmbd_vals):
                nn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                                    alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
                nn.fit(self.X_train, self.y_train)
                
                self.NN_list[i][j] = nn
                
                print("Learning rate  = ", eta)
                print("Lambda = ", lmbd)
                print("Accuracy score on test set: ", nn.score(self.X_test, self.y_test))
                print()
    
    def NN_visualization(self):
        sns.set()

        train_accuracy = np.zeros((self.n_eta, self.n_lmbd))
        test_accuracy = np.zeros((self.n_eta, self.n_lmbd))

        for i in range(self.n_eta):
            for j in range(self.n_lmbd):
                nn = self.NN_list[i][j]
                
                train_pred = nn.predict(self.X_train) 
                test_pred = nn.predict(self.X_test)

                train_accuracy[i][j] = accuracy_score(self.y_train, train_pred)
                test_accuracy[i][j] = accuracy_score(self.y_test, test_pred)

                
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()

        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()
    
    




