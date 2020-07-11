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

        categorical = ["chess", "fungi", "numbers", "tic-tac-toe"]
        continuous = ["banknote", "ecoli", "glass", "iris", "libras", "wine"]
        self.paths = categorical + continuous

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
        ohe = preprocessing.OneHotEncoder()
        oe = preprocessing.OrdinalEncoder()
        self.dataset.data = ohe.fit_transform(self.dataset.data).toarray()
        self.dataset.target = oe.fit_transform(
            self.dataset.target.reshape(-1, 1))
        self.dataset.target = self.dataset.target.reshape(1, -1)[0]

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
        print(
            "=======================================================================")
        print(self.paths[self.datasetIndex])
        print(
            "=======================================================================")
        for classifier in self.predicted_classes.keys():
            print(
                "=======================================================================")
            print(self.paths[self.datasetIndex])
            print("Resultados do classificador: %s\n%s\n"
                  % (classifier, metrics.classification_report(self.dataset.target, self.predicted_classes[classifier])))
            print("Matriz de confusão: \n%s\n\n\n" % metrics.confusion_matrix(
                self.dataset.target, self.predicted_classes[classifier]))

    def saveInfo(self):
        for classifier in self.predicted_classes.keys():
            report = metrics.classification_report(
                self.dataset.target, self.predicted_classes[classifier], output_dict=True)
            dfReport = pd.DataFrame(data=report).transpose()
            dfReport.to_csv("Results/%s/%s.csv" %
                            (self.paths[self.datasetIndex], classifier))

    def default_rotine(self):
        self.initAlgorithm()
        self.startTraining()
        self.showInfo()
        self.saveInfo()


if __name__ == '__main__':
    for i in range(0, 10):
        comparador = Trabalho(
            dataFilePath="C:/Users/gabre/Documents/IA/Trabalho2/Datasets/%s/%s.csv", datasetIndex=i)
        comparador.default_rotine()
