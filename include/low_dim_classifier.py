import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from include.cnf_matrix_pct import compute_cnf_matrix



class Low_dim_classifier:
    def __init__(self,x,y,methods=['SVC']):
        self.x=x
        self.y=y
        if not isinstance(methods, list):
            methods = [methods]
        self.method=methods
        self.sclr = StandardScaler()
        self.X = self.sclr.fit_transform(x)
        self.get_classifiers(methods)


    def get_classifiers(self,methods):
        classifiers = {}
        self.available_classifiers=[]
        for method in methods:
            if method == 'KNeighborsClassifier' or method == 'all':
                classifiers['KNeighborsClassifier']=KNeighborsClassifier(3)
                self.available_classifiers.append('KNeighborsClassifier')

            if method == 'SVC_linear' or method == 'all':
                classifiers['SVC_linear']=SVC(kernel="linear", C=0.025)
                self.available_classifiers.append('SVC_linear')

            if method == 'SVC' or method == 'all':
                classifiers['SVC']=SVC(gamma=2, C=1)
                self.available_classifiers.append('SVC')

            if method == 'GaussianProcessClassifier' or method == 'all':
                classifiers['GaussianProcessClassifier']=GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
                self.available_classifiers.append('GaussianProcessClassifier')

            if method == 'DecisionTreeClassifier' or method == 'all':
                classifiers['DecisionTreeClassifier']=DecisionTreeClassifier(max_depth=5)
                self.available_classifiers.append('DecisionTreeClassifier')

            if method == 'RandomForestClassifier' or method == 'all':
                classifiers['RandomForestClassifier']=RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)
                self.available_classifiers.append('RandomForestClassifier')

            if method == 'MLPClassifier' or method == 'all':
                classifiers['MLPClassifier']=MLPClassifier(alpha=1)
                self.available_classifiers.append('MLPClassifier')

            if method == 'AdaBoostClassifier' or method == 'all':
                classifiers['AdaBoostClassifier']=AdaBoostClassifier()
                self.available_classifiers.append('AdaBoostClassifier')

            if method == 'GaussianNB' or method == 'all':
                classifiers['GaussianNB']=GaussianNB()
                self.available_classifiers.append('GaussianNB')

            if method == 'QuadraticDiscriminantAnalysis' or method == 'all':
                classifiers['QuadraticDiscriminantAnalysis']=QuadraticDiscriminantAnalysis()
                self.available_classifiers.append('QuadraticDiscriminantAnalysis')

        self.classifiers=classifiers

    def train(self,train_methods=['SVC']):
        if not isinstance(train_methods, list):
            train_methods = [train_methods]
        if train_methods==['all']:
            train_methods=self.available_classifiers


        for method in train_methods:
            print 'Training '+ method
            self.classifiers[method].fit(self.X,self.y)

    def predict(self,x_test,test_methods=['SVC']):
        if not isinstance(test_methods, list):
            test_methods = [test_methods]
        X_test=self.sclr.transform(x_test)
        y_pred={}
        for method in test_methods:
            print 'Predicting with '+ method
            y_pred[method]=self.classifiers[method].predict(X_test)

        return y_pred

    def get_performance(self,x_test,y_test,perf_methods=['SVC']):
        if not isinstance(perf_methods, list):
            perf_methods = [perf_methods]

        y_pred = self.predict(x_test,perf_methods)
        cnf_mats={}
        for method in perf_methods:
            print 'Computing Confusion Matrix for '+method
            cnf_matrix, cnf_matrix_pct = compute_cnf_matrix(y_test, y_pred[method])
            cnf_mats[method]=np.vstack([cnf_matrix, cnf_matrix_pct])
        return y_pred,cnf_mats



















