import random as rng
import numpy as np

class AlocacaoArtigos:
    
    def __init__ (self, dataFilePath):
        
        self.dataFilePath = dataFilePath
        self.loadDataFromFile()
        
    def loadDataFromFile(self):
        dataSetLoded = np.loadtxt(self.dataFilePath, delimiter=',')
        self.nArtigos = len(dataSetLoded[0]) - 1
        self.individuos = []
        #TODO refatorar para codigo mais bonito
        for i in dataSetLoded:
            artigos = []
            capacidade = 0
            for j in range(len(i)):
                if j < self.nArtigos:
                    artigos.append(i[j])
                else:
                    capacidade = i[j]
            self.individuos.append([artigos, capacidade])
        
    
    def run(self):
        print('running')
        
    
    
    
if __name__ == '__main__':
    geneticAlg = AlocacaoArtigos("C:/Users/gabre/Documents/IA/Trabalho/database.txt")
    geneticAlg.run()