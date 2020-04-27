import random
import numpy as np
import matplotlib.pyplot as plot

class LogisticRegression:
    def __init__(self, dataFilePath, outputPath, alpha=0.01, maxIter=500, threshold=0.5, errorThreshold=0.001):
        self.dataFilePath = dataFilePath
        self.outputPath = outputPath
        self.alpha = alpha
        self.maxIter = maxIter
        self.threshold = threshold
        self.errorThreshold = errorThreshold

        self.loadDataFromFile()
        self.initWeights()

    def loadDataFromFile(self):

    def initWeights(self):

    