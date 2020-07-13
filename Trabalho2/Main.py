from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import datetime


class dataset():
    def __init__(self, data, target):
        self.target = target
        self.data = data


class Trabalho:

    def __init__(self, dataFilePath=None, outputPath=None, datasetIndex=1, testDataSet=False, maxIter=100):
        self.dataFilePath = dataFilePath
        self.outputPath = outputPath
        self.testDataSet = testDataSet
        if (dataFilePath is None) and (testDataSet == False):
            self.testDataSet = True
        self.maxIter = maxIter
        self.datasetIndex = datasetIndex

        self.categorical = ["chess", "fungi", "numbers", "tic-tac-toe"]
        self.continuous = ["banknote", "ecoli",
                           "glass", "iris", "libras", "wine"]
        self.paths = self.categorical + self.continuous

        self.loadDataset()
        self.initAlgorithm()

    def loadDataset(self):
        index = self.datasetIndex
        if self.testDataSet == False:
            ds = pd.read_csv(self.dataFilePath %
                             (self.paths[index], self.paths[index]))
            ds = np.array(ds)

            data = []
            target = []
            for row in range(ds.shape[0]):
                a = []
                for col in range(ds.shape[1]-1):
                    a.append(ds[row][col])
                a = np.array(a)
                data.append(a)
                target.append(ds[row][-1])
            data = np.array(data)
            target = np.array(target)

            self.dataset = dataset(data, target)
        else:
            self.dataset = datasets.load_digits()

    def initAlgorithm(self):
        self.dt = tree.DecisionTreeClassifier()
        self.knn = KNeighborsClassifier()
        self.nb = GaussianNB()
        self.mlp = MLPClassifier()
        self.logReg = LogisticRegression()

    def startTraining(self):
        ohe = preprocessing.OneHotEncoder()
        oe = preprocessing.OrdinalEncoder()

        if(self.paths[self.datasetIndex] in self.categorical):
            self.dataset.data = ohe.fit_transform(self.dataset.data).toarray()

        self.dataset.target = oe.fit_transform(
            self.dataset.target.reshape(-1, 1))
        self.dataset.target = self.dataset.target.reshape(1, -1)[0]

        Scoring = ['accuracy', 'f1_macro']

        parametres_tree = {'splitter': (
            'best', 'random'), 'max_depth': [4, 8, 16, None]}
        self.gsTree = model_selection.GridSearchCV(
            self.dt, parametres_tree, return_train_score=True, cv=10, scoring=Scoring, refit='accuracy')

        parametres_knn = {'weights': ('uniform', 'distance'), 'p': (1, 2)}
        self.gsKnn = model_selection.GridSearchCV(
            self.knn, parametres_knn, return_train_score=True, cv=10, scoring=Scoring, refit='accuracy')

        parametres_naiveb = {'var_smoothing': [1e-9, 1e-10]}
        self.gsNaiveb = model_selection.GridSearchCV(
            self.nb, parametres_naiveb, return_train_score=True, cv=10, scoring=Scoring, refit='accuracy')

        parametres_mlp = {'hidden_layer_sizes': [100, 200], 'learning_rate_init': [
            0.001, 0.01], 'max_iter': [200, 300], 'alpha': [0.0001, 0.001]}
        self.gsMlp = model_selection.GridSearchCV(
            self.mlp, parametres_mlp, return_train_score=True, cv=10, scoring=Scoring, refit='accuracy')

        parametres_logreg = {'C': [1.0, 2.0], 'solver': (
            'newton-cg', 'lbfgs', 'sag', 'saga'), 'max_iter': [100, 200, 300]}
        self.gsLogreg = model_selection.GridSearchCV(
            self.logReg, parametres_logreg, return_train_score=True, cv=10, scoring=Scoring, refit='accuracy')

        self.gsTree.fit(self.dataset.data, self.dataset.target)
        self.gsKnn.fit(self.dataset.data, self.dataset.target)
        self.gsNaiveb.fit(self.dataset.data, self.dataset.target)
        self.gsMlp.fit(self.dataset.data, self.dataset.target)
        self.gsLogreg.fit(self.dataset.data, self.dataset.target)

    def showInfo(self):
        print("=======================================================================")
        print(self.paths[self.datasetIndex])
        print("=======================================================================")
        print("-----------------------------------------------------------------------")
        print("Tree")
        print("////////////")
        print(self.gsTree.best_estimator_)
        print("////////////")
        print(self.gsTree.best_score_)
        print("////////////")
        print(self.gsTree.best_params_)
        print("////////////")
        print(self.gsTree.best_index_)
        print("////////////")
        print(self.gsTree.scorer_)
        print("-----------------------------------------------------------------------")
        print("Knn")
        print("////////////")
        print(self.gsKnn.best_estimator_)
        print("////////////")
        print(self.gsKnn.best_score_)
        print("////////////")
        print(self.gsKnn.best_params_)
        print("////////////")
        print(self.gsKnn.best_index_)
        print("////////////")
        print(self.gsKnn.scorer_)
        print("-----------------------------------------------------------------------")
        print("Naive Bayes")
        print("////////////")
        print(self.gsNaiveb.best_estimator_)
        print("////////////")
        print(self.gsNaiveb.best_score_)
        print("////////////")
        print(self.gsNaiveb.best_params_)
        print("////////////")
        print(self.gsNaiveb.best_index_)
        print("////////////")
        print(self.gsNaiveb.scorer_)
        print("-----------------------------------------------------------------------")
        print("MLP")
        print("////////////")
        print(self.gsMlp.best_estimator_)
        print("////////////")
        print(self.gsMlp.best_score_)
        print("////////////")
        print(self.gsMlp.best_params_)
        print("////////////")
        print(self.gsMlp.best_index_)
        print("////////////")
        print(self.gsMlp.scorer_)
        print("-----------------------------------------------------------------------")
        print("Logistic Regration")
        print("////////////")
        print(self.gsLogreg.best_estimator_)
        print("////////////")
        print(self.gsLogreg.best_score_)
        print("////////////")
        print(self.gsLogreg.best_params_)
        print("////////////")
        print(self.gsLogreg.best_index_)
        print("////////////")
        print(self.gsLogreg.scorer_)

    def saveInfo(self):
        dtReport = pd.DataFrame.from_dict(data=self.gsTree.cv_results_)
        dtReport.to_csv("Results/%s/Tree.csv" %
                        (self.paths[self.datasetIndex]))

        dtReport = pd.DataFrame.from_dict(data=self.gsKnn.cv_results_)
        dtReport.to_csv("Results/%s/Knn.csv" %
                        (self.paths[self.datasetIndex]))

        dtReport = pd.DataFrame.from_dict(data=self.gsNaiveb.cv_results_)
        dtReport.to_csv("Results/%s/NaiveB.csv" %
                        (self.paths[self.datasetIndex]))

        dtReport = pd.DataFrame.from_dict(data=self.gsMlp.cv_results_)
        dtReport.to_csv("Results/%s/MLP.csv" %
                        (self.paths[self.datasetIndex]))

        dtReport = pd.DataFrame.from_dict(data=self.gsLogreg.cv_results_)
        dtReport.to_csv("Results/%s/LogReg.csv" %
                        (self.paths[self.datasetIndex]))

    def default_rotine(self):
        self.initAlgorithm()
        self.startTraining()
        self.saveInfo()
        self.showInfo()


if __name__ == '__main__':
    start = datetime.now()
    comparador = Trabalho(
        dataFilePath="Datasets/%s/%s.csv", datasetIndex=9)
    comparador.default_rotine()
    print(datetime.now()-start)
