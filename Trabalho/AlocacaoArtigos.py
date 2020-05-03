import random as rng
import numpy as np


class AlocacaoArtigos:

    def __init__(self, dataFilePath):

        self.dataFilePath = dataFilePath
        self.loadDataFromFile()

    def loadDataFromFile(self):
        dataSetLoded = np.loadtxt(self.dataFilePath, delimiter=',')
        self.nArtigos = len(dataSetLoded[0]) - 1
        self.matrizReferencia = []
        # TODO refatorar para codigo mais bonito
        for i in dataSetLoded:
            artigos = []
            capacidade = 0
            for j in range(len(i)):
                if j < self.nArtigos:
                    artigos.append(i[j])
                else:
                    capacidade = i[j]
            self.matrizReferencia.append([artigos, capacidade])
        self.fitnessTotal = []

    def run(self):
        print('running')
        # print(self.individuos)  # matriz para referencia de afinidade
        # print(self.individuos[3])  # recuperando um individuo na matriz

        teste = []
        teste.append([1, 1, 1, 1, 1])
        teste.append([1, 1, 0, 1, 0])
        teste.append([1, 0, 1, 1, 0])
        teste.append([0, 0, 1, 1, 0])

        teste2 = []
        teste2.append([0, 1, 1, 0, 0])
        teste2.append([1, 1, 1, 1, 0])
        teste2.append([1, 1, 0, 1, 0])
        teste2.append([1, 1, 0, 1, 0])

    # gera a população inicial aleatoriamente

    def initialize(self):
        populationSize = 10
        numRevisores = 4
        numArtigos = 5
        for n in range(populationSize):  # gerar 10 individuos
            # enquanto não houver individuos validos suficientes
            while len(self.populacao) < populationSize:
                novoIndividuo = []

                for i in range(numRevisores):  # para cada revisor necessário
                    revisor = [0] * 5  # novo revisor
                    limiteDeArtigos = int(self.matrizReferencia[i][1])

                    # repetindo suficiente para usar o limite
                    for j in range(limiteDeArtigos):
                        position = random.randint(0, 4)
                        revisor[position] = 1

                    novoIndividuo.append(revisor)
                if self.validate(novoIndividuo):
                    self.populacao.append(novoIndividuo)
        return

    def showPopulation(self):
        for i in range(len(self.populacao)):
            print('\nINDIVIDUO:', i+1)
            for j in self.populacao[i]:
                print(j)

    # valida um individuo de acordo com seus limites e distribuição de artigos
    def validate(self, individuo):
        artigosAlocados = [False] * 5

        for i in range(len(individuo)):
            revisor = individuo[i]

            contagemArtigos = 0
            limiteDeArtigos = int(self.matrizReferencia[i][1])
            for artigo in range(len(revisor)):
                if revisor[artigo] == 1:
                    contagemArtigos += 1
                    # caso o artigo já tenha sido dado a um revisor
                    if artigosAlocados[artigo] == True:
                        return False
                    else:
                        artigosAlocados[artigo] = True

                if contagemArtigos > limiteDeArtigos:  # caso o artigo exceda o limite de artigos do revisor
                    return False

        if False in artigosAlocados:  # se algum artigo não tiver sido alocado
            return False
        return True

    # causa mutação em um individuo, descartando resultados inválidos
    def mutate(self, individuo):
        return

    # mescla parte de um individuo com parte de outro, descartando resultados invalidos
    def reproduce(self, individuox, individuoy):
        corte = rng.randint(0, len(self.matrizReferencia) - 1)
        print(corte)
        novox = []
        novoy = []
        resultado = []

        for i in range(0, len(self.matrizReferencia)):
            if i < corte:
                novox.append(individuox[i])
                novoy.append(individuoy[i])
            else:
                novox.append(individuoy[i])
                novoy.append(individuox[i])

        resultado.append([novox, novoy])
        return resultado

    # seleciona os individuos mais adaptados para a próxima geração
    def selection(self, individuos):
        self.fitnessTotal.clear()

        for individuo in individuos:
            self.fitnessTotal.append(self.fitness(individuo))

        total = np.sum(self.fitnessTotal)
        porcaoRoleta = []

        for fitness in self.fitnessTotal:
            porcaoRoleta.append((fitness * 360) / total)
        for i in range(0, len(porcaoRoleta)):
            if (i > 0):
                porcaoRoleta[i] += porcaoRoleta[i - 1]

        selecionados = []
        for rodadas in range(1, len(self.fitnessTotal) - 1):
            rnga = rng.randint(0, 359)
            rngb = rng.randint(0, 359)
            for i in range(0, len(porcaoRoleta)):
                individuoa = -1
                individuob = -1
                if (i == 0) and (rnga < porcaoRoleta[i]):
                    individuoa = i
                if (i == 0) and (rngb < porcaoRoleta[i]):
                    individuob = i
                if (i > 0) and (porcaoRoleta[i - 1] < rnga) and (rnga < (porcaoRoleta[i])):
                    individuoa = i
                if (i > 0) and (porcaoRoleta[i - 1] < rngb) and (rngb < (porcaoRoleta[i])):
                    individuob = i
            selecionados.append[individuos[individuoa], individuos[individuob]]

        return selecionados

    # Função fitness do individuo, baseada na média de afinidade entre os revisores e artigos
    def fitness(self, individuo):
        notasAvaliadas = []
        for i in range(0, len(individuo)):
            for j in range(0, len(individuo[i])):
                if individuo[i][j] == 1:
                    notasAvaliadas.append(self.matrizReferencia[i][0][j])
        return np.mean(notasAvaliadas)

    # função principal, aloca revisores e artigos usando mutação, reprodução e seleção.
    def alocate(self, population):
        return

    # percorre um corretor e conta quantos artigos ele está alocado para corrigir
    def contAloc(self, corretor):
        count = 0
        for art in corretor:
            if art == 1:
                count += 1
        return count

    def grafico(self):
        return

    def resultadoTotxt(self):
        return


if __name__ == '__main__':
    geneticAlg = AlocacaoArtigos(
        "C:/Users/gabre/Documents/IA/Trabalho/database.txt")
    geneticAlg.run()
