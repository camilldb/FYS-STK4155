from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from imblearn.over_sampling import SMOTE

class ClassificationTree:
    def __init__(self, X, y):
        #Feature matrix, must be a pd.DataFrame with column names = feature names
        self.X = X
        self.y = y
        #smote sampling
        #os = SMOTE(random_state = 0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        
        #self.X_train, self.y_train = os.fit_resample(self.X_train, self.y_train)

    def makeTree(self):
        self.tree_clf = DecisionTreeClassifier(max_depth = 5)
        self.tree_clf.fit(self.X_train, self.y_train)
    
    def plotTree(self):
        tree.plot_tree(self.tree_clf)
    
    def printScore(self):
        test_pred = self.tree_clf.predict(self.X_test)
        train_pred = self.tree_clf.predict(self.X_train)
        print("Test set accuracy with Decision Trees: {:.2f}".format(test_pred))
        print("Train set accuracy with Decision Trees: {:.2f}".format(train_pred))

    def doAll(self):
        self.makeTree()
        self.plotTree()
        self.printScore()