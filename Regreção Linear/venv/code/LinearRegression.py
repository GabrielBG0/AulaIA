import numpy as np
import random
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, dataFilePath, outputPath, alpha=0.01,
                 maxIter=500, errorThreshold=0.001,
                 performTest=False, normalize=False):

        self.dataFilePath = dataFilePath
        self.outputPath = outputPath
        self.alpha = alpha
        self.maxIter = maxIter
        self.errorThreshold = errorThreshold
        self.performTest = performTest
        self.normalize = normalize

        self.loadDataFromFile()
        self.initWeights()

    def featureNormalize(self, X):
        X_Norm = X
        for i in range(len(x[0])):
            m = np.mean(x[:,i])
            s = np.std(x[:,i])
            X_norm[:,i] = (X_Norm[:, i] - m) /s
        return X_Norm

    def loadDataFromFile(self):
        datesettloaded = np.loadtxt(self.dataFilePath, delimiter=";", skiprows=1)

        if self.normalize:
            datesettloaded = self.featureNormalize(datesettloaded)

        self.nExemples = datesettloaded.shape[0]
        self.nAttributes = len(datesettloaded[0])

        if self.performTest:
            nExemplesTest = int(self.nExemples/3.0)
            self.testData = np.ones(shape=(nExemplesTest, self.nAttributes))
            self.testTarget = np.zeros(shape=(nExemplesTest,1))

            linesForTest = random.sample(range(0, self.nExemples), nExemplesTest)

            count = 0
            for line in linesForTest:
                self.testData[count,1] =  datesettloaded[line, :-1]
                self.testTarget[count] = datesettloaded[line, -1]
                count += 1
            datesettloaded = np.delete(datesettloaded, linesForTest, 0)
            self.nExemples -= nExemplesTest
        
        self.dataset = np.ones(shape=(self.nExemples, self.nAttributes))
        self.dataset[:,1:] = datesettloaded[:, :-1]
        self.target = datesettloaded[:, -1]
        self.target.shape = (self.nExemples, 1)


    def initWeights(self):
        # TODO: INICIAR OS PESOS: THETA0 e THETA1
        self.weigths = np.ones(shape=(self.nAttributes, 1))
        for i in range(0,self.nAttributes):
            self.weigths[i][0] = random.random()

    def linearFunction(self, data):
        # TODO: SAIDA DA FUNCAO LINEAR = THETA(t) * X
        output = data.dot(self.weigths)
        return output

    def calculateError(self, data, target):
        # TODO: CALCULAR O ERRO PARA UM PONTO
        output = self.linearFunction(data)
        error = output - target
        return error


    def squaredErrorCost(self, data, target):
        # TODO: CALCULAR O ERRO PARA TODOS OS PONTOS
        error = self.calculateError(data, target)
        squaredError = (1.0 / (2 * nExemples)) * (error.T.dot(error))


    def gradientDescent(self):
        # TODO: GRADIENTE DESCENDENTE


    def plotCostGraph(self, trainingErrorsList, testingErrorsList=None):
        xAxisValues = range(0, len(trainingErrorsList))
        line1 = plt.plot(xAxisValues, trainingErrorsList, label="Training error")
        if self.performTest:
            line2 = plt.plot(xAxisValues, testingErrorsList, linestyle="dashed", label="Testing error")

        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("LMS Error")
        plt.savefig(self.outputPath + "/lms_error.png")
        plt.show()
        plt.close()

    def plotLineGraph(self, weightsToPlot, iteration):

        if self.performTest:
            dataToPlot = np.append(self.dataset, self.testData, 0)
            targetToPlot = np.append(self.target, self.testTarget, 0)

        else:
            dataToPlot = self.dataset
            targetToPlot = self.target

        xAxisValues = dataToPlot[:, 1]
        yAxisValues = targetToPlot

        xMax = max(xAxisValues)
        xMin = min(xAxisValues)
        yMax = max(yAxisValues)

        axes = plt.gca()
        axes.set_xlim([0, xMax + 1])
        axes.set_ylim([0, yMax + 1])

        xLineValues = np.arange(xMin, xMax, 0.1)
        yLineValues = weightsToPlot[0] + xLineValues * weightsToPlot[1]

        plt.plot(xLineValues, yLineValues)
        plt.plot(xAxisValues, yAxisValues, 'o')
        plt.savefig(self.outputPath + "/line_" + str(iteration) + ".png")
        plt.close()

    def run(self):
        # TODO: PRINCIPAL

if __name__ == '__main__':
    linReg = LinearRegression("D:/Nextcloud/UFMS/Aulas/2020-1/IA/codigos/income/income.csv",
                              "D:/Nextcloud/UFMS/Aulas/2020-1/IA/codigos/income",
                              normalize=True, performTest=True, alpha=0.0001, maxIter=1000)
    linReg.run()