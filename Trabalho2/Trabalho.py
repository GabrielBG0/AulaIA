from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from sklearn import datasets
import numpy as np


class Trabalho:

    def __init__(self, dataFilePath=None, outputPath=None, testDataSet=False, maxIter=100, fileType=''):
        self.dataFilePath = dataFilePath
        self.outputPath = outputPath
        self.testDataSet = testDataSet
        if (dataFilePath == None) and (testDataSet == False):
            self.testDataSet = True
        self.maxIter = maxIter
        self.loadDataset()
        self.initAlgorithm()

    def loadDataset(self):
        if self.testDataSet:
            self.dataset = datasets.load_digits()
        else:
            return None

    def initAlgorithm(self):
        self.dt = tree.DecisionTreeClassifier()
        self.knn = KNeighborsClassifier()
        self.nb = GaussianNB()
        self.mlp = MLPClassifier()
        self.logReg = LogisticRegression()
        self.kf = model_selection.StratifiedKFold(n_splits=10)

        self.predicted_classes = dict()
        self.predicted_classes['tree'] = np.zeros(self.dataset.target.shape[0])
        self.predicted_classes['knn'] = np.zeros(self.dataset.target.shape[0])
        self.predicted_classes['naiveb'] = np.zeros(
            self.dataset.target.shape[0])
        self.predicted_classes['mlp'] = np.zeros(self.dataset.target.shape[0])
        self.predicted_classes['logReg'] = np.zeros(
            self.dataset.target.shape[0])

    def startTraining(self):
        for train, test in self.kf.split(self.dataset.data, self.dataset.target):
            data_train, target_train = self.dataset.data[train], self.dataset.target[train]
            data_test, target_test = self.dataset.data[test], self.dataset.target[test]

            self.dt = self.dt.fit(data_train, target_train)
            dt_predicted = self.dt.predict(data_test)
            self.predicted_classes['tree'][test] = dt_predicted

            self.knn = self.knn.fit(data_train, target_train)
            knn_predicted = self.knn.predict(data_test)
            self.predicted_classes['knn'][test] = knn_predicted

            self.nb = self.nb.fit(data_train, target_train)
            nb_predicted = self.nb.predict(data_test)
            self.predicted_classes['naiveb'][test] = nb_predicted

            self.mlp = self.mlp.fit(data_train, target_train)
            mlp_predicted = self.mlp.predict(data_test)
            self.predicted_classes['mlp'][test] = mlp_predicted

            self.logReg = self.logReg.fit(data_train, target_train)
            logReg_predicted = self.logReg.predict(data_test)
            self.predicted_classes['logReg'][test] = logReg_predicted

    def showInfo(self):
        for classifier in self.predicted_classes.keys():
            print(
                "=======================================================================")
            print("Resultados do classificador: %s\n%s\n"
                  % (classifier, metrics.classification_report(self.dataset.target, self.predicted_classes[classifier])))
            print("Matriz de confusão: \n%s\n\n\n" % metrics.confusion_matrix(
                self.dataset.target, self.predicted_classes[classifier]))

    def default_rotine(self):
        self.initAlgorithm()
        self.startTraining()
        self.showInfo()


if __name__ == '__main__':
    comparador = Trabalho(testDataSet=True)
    comparador.default_rotine()
