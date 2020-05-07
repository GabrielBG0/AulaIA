import random as rng
import numpy as np


class AlocacaoArtigos:

    populationSize = 6
    numRevisores = 4
    numArtigos = 5
    maxgen = 100
    mutationrate = 0.1
    crossoverrate = 0.7

    def __init__(self, dataFilePath, outputPath):

        self.dataFilePath = dataFilePath
        self.outputPath = outputPath
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
        self.historicoFitness = []
        self.populacao = []

    def run(self):
        print('running')
        teste = []
        teste.append([1, 1, 1, 1, 1])
        teste.append([0, 0, 0, 0, 0])
        teste.append([1, 0, 1, 1, 0])
        teste.append([0, 0, 1, 1, 0])

        teste2 = []
        teste2.append([0, 1, 1, 0, 0])
        teste2.append([1, 1, 1, 1, 0])
        teste2.append([1, 1, 0, 1, 0])
        teste2.append([1, 1, 0, 1, 0])

        teste3 = []
        teste3.append([0, 0, 1, 0, 0])
        teste3.append([1, 1, 0, 0, 0])
        teste3.append([0, 0, 0, 1, 0])
        teste3.append([0, 0, 0, 0, 1])

        self.resultadoTotxt(teste3)

    # gera a população inicial aleatoriamente
    def initialize(self):

        for n in range(self.populationSize):  # gerar 6 individuos
            # enquanto não houver individuos validos suficientes
            while len(self.populacao) < self.populationSize:
                novoIndividuo = []

                for i in range(self.numRevisores):  # para cada revisor necessário
                    revisor = [0] * 5  # novo revisor
                    limiteDeArtigos = int(self.matrizReferencia[i][1])

                    # repetindo suficiente para usar o limite
                    for j in range(limiteDeArtigos):
                        position = rng.randint(0, 4)
                        revisor[position] = 1

                    novoIndividuo.append(revisor)
                if self.validate(novoIndividuo):
                    self.populacao.append(novoIndividuo)
        self.showPopulation()
        return

    def showPopulation(self):
        for i in range(len(self.populacao)):
            print('\nINDIVIDUO:', i+1, 'FITNESS: ',
                  self.fitness(self.populacao[i]))
            for j in self.populacao[i]:
                print(j)

    # valida um individuo de acordo com seus limites e distribuição de artigos
    def validate(self, individuo):
        artigosAlocados = [False] * self.numArtigos

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
        valido = False
        while valido == False:
            mutatedRow = [0] * 5
            revisorEscolhido = rng.randint(0, len(individuo)-1)
            limite = int(self.matrizReferencia[revisorEscolhido][1])
            for i in range(limite):
                posicao = rng.randint(0, len(individuo[revisorEscolhido]) - 1)
                mutatedRow[posicao] = 1
            individuo[revisorEscolhido] = mutatedRow

            valido = self.validate(individuo)
        # individuo = self.correcao(individuo)

        return individuo

    # mescla parte de um individuo com parte de outro, descartando resultados invalidos
    def reproduce(self, individuox, individuoy):
        corte = rng.randint(0, len(self.matrizReferencia) - 1)
        # print(corte)
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

    def correcao(self, individuo):

        for i in range(0, len(self.matrizReferencia)):
            aloc = self.contAloc(individuo[i])
            while aloc > self.matrizReferencia[i][1]:
                j = rng.randint(0, len(individuo[i]) - 1)
                if individuo[i][j] == 1:
                    individuo[i][j] = 0
                    aloc -= 1
        for i in range(0, len(individuo)):
            corretoresx = []
            for j in range(0, self.nArtigos - 1):
                if individuo[j][i] == 1:
                    corretoresx.append(j)
            if len(corretoresx) > 1:
                maiorNota = 0
                indiceCorretor = -1
                for corretor in range(0, len(corretoresx)):
                    if corretor == 0:
                        maiorNota = self.matrizReferencia[corretoresx[corretor]][0][i]
                        indiceCorretor = corretoresx[corretor]
                    elif self.matrizReferencia[corretoresx[corretor]][0][i] > maiorNota:
                        individuo[indiceCorretor][i] = 0
                        maiorNota = self.matrizReferencia[corretoresx[corretor]][0][i]
                        indiceCorretor = corretoresx[corretor]
                    else:
                        individuo[corretoresx[corretor]][i] = 0
        semCorretor = []
        for i in range(0, self.nArtigos):
            corretoresx = []
            for j in range(0, self.nArtigos - 1):
                if individuo[j][i] == 1:
                    corretoresx.append(j)
            if len(corretoresx) == 0:
                semCorretor.append(i)
        for corretor in range(0, len(individuo)):
            if len(semCorretor) > 0:
                corrigidos = 0
                for artigo in range(0, self.nArtigos):
                    if individuo[corretor][artigo] == 1:
                        corrigidos += 1
                if corrigidos < self.matrizReferencia[corretor][1]:
                    individuo[corretor][semCorretor.pop()] = 1
            else:
                break

        return individuo

    def crossover(self, individuox, individuoy):
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

        novox = self.correcao(novox)
        novoy = self.correcao(novoy)

        resultado.append([novox, novoy])
        return resultado

    # seleciona os individuos mais adaptados para a próxima geração
    def selection(self, individuos):
        self.fitnessTotal.clear()

        for individuo in individuos:
            self.fitnessTotal.append(self.fitness(individuo))

        total = np.sum(self.fitnessTotal)
        porcaoRoleta = []

        self.historicoFitness.append(np.mean(self.fitnessTotal))

        for fitness in self.fitnessTotal:
            porcaoRoleta.append((fitness * 360) / total)
        for i in range(0, len(porcaoRoleta)):
            if (i > 0):
                porcaoRoleta[i] += porcaoRoleta[i - 1]

        selecionados = []
        for rodadas in range(0, len(self.fitnessTotal) // 2):
            rnga = rng.randint(0, 359)
            rngb = rng.randint(0, 359)
            individuoa = -1
            individuob = -1
            for i in range(0, len(porcaoRoleta)):
                if (i == 0) and (rnga < porcaoRoleta[i]):
                    individuoa = i
                if (i == 0) and (rngb < porcaoRoleta[i]):
                    individuob = i
                if (i > 0) and (porcaoRoleta[i - 1] < rnga) and (rnga < (porcaoRoleta[i])):
                    individuoa = i
                if (i > 0) and (porcaoRoleta[i - 1] < rngb) and (rngb < (porcaoRoleta[i])):
                    individuob = i
            selecionados.append(
                [individuos[individuoa], individuos[individuob]])

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
        for geracao in range(self.maxgen):  # numero de gerações
            newPopulation = []
            selecionados = self.selection(self.populacao)
            for couple in range(len(selecionados)):
                parents = selecionados[couple]
                crossoverChance = rng.random()
                children = []
                if crossoverChance < self.crossoverrate:
                    children = self.crossover(parents[0], parents[1])
                else:
                    children.append(parents)
                for child in children[0]:
                    mutationChance = rng.random()
                    if mutationChance < self.mutationrate:
                        child = self.mutate(child)
                    newPopulation.append(child)
            self.populacao = newPopulation
            print('\nGERAÇÃO #', geracao+1)
            self.showPopulation()

        return

    # percorre um corretor e conta quantos artigos ele está alocado para corrigir
    def contAloc(self, corretor):
        count = 0
        for art in corretor:
            if art == 1:
                count += 1
        return count

    def plotFitnessGraph(self, bestTrainingErrorsList, avredgeTraningListError):
        xAxisValues = range(0, len(trainingErrorsList))

        line1 = plt.plot(xAxisValues, trainingErrorsList, label="Best Fitness")
        line2 = plt.plot(xAxisValues, avredgeTraningListError,
                         linestyle="dashed", label="Avredge Traning List Error")

        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.savefig(self.outputPath + "/fitness.png")
        plt.show()
        plt.close()

    def resultadoTotxt(self, individuo):
        result = []
        for artigo in range(0, self.nArtigos):
            for corretor in range(0, len(self.matrizReferencia)):
                if individuo[corretor][artigo] == 1:
                    result.append(corretor)
                    break

        np.savetxt('saida-genetico.txt', result, fmt='%d',
                   newline=',')


if __name__ == '__main__':
    geneticAlg = AlocacaoArtigos(
        "C:/Users/gabre/Documents/IA/Trabalho/database.txt", "C:/Users/gabre/Documents/IA/Trabalho")
    geneticAlg.run()
