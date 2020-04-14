import random


class Objetos:
    def __init__(self, peso, valor):
        self.peso = peso
        self.valor = valor

    def getPeso(self):
        return self.peso

    def getValor(self):
        return self.valor


lista_objetos = []
lista_objetos.append(objetos(4, 2))
lista_objetos.append(objetos(3, 7))
lista_objetos.append(objetos(2, 5))
lista_objetos.append(objetos(1, 1))
lista_objetos.append(objetos(3, 9))
lista_objetos.append(objetos(1, 2))
lista_objetos.append(objetos(5, 3))
lista_objetos.append(objetos(4, 4))
lista_objetos.append(objetos(4, 4))
lista_objetos.append(objetos(3, 7))


populacaoInicial = []

class Individuo(organizacao, fitness):
    def __init__(self, orgnizacao, fitness):
        self.organizacao = orgnizacao
        self.fitness = fitness
    def getOrganizacao(self):
        return self.organizacao
    def getFitness(self):
        return self.fitness

def Fitness(peso, valor):
    return valor/peso


def inicia_populacao(numero_individuos, tamanho_mochila):
    for i in range(0, numero_individuos):
        organizacao = set()
        peso = 0
        while peso < tamanho_mochila:
            objeto = lista_objetos[random.randint(0, len(lista_objetos))]
            if objeto.getPeso() + peso < tamanho_mochila:
                organizacao.add(objeto)
            else:
                break
            for i in organizacao:
                peso += objeto.getPeso()
        fitness = 0
        for j in organizacao:
            fitness += Fitness(organizacao[i].getPeso(), organizacao[i].getValor())

        populacaoInicial.append(Individuo(organizacao,fitness))

def reproduzir (objetoX, objetoY):
    tamanhox = len(objetoX.getOrganizacao)
    tamanhoy = len(objetoY.getOrganizacao)
    organizacaoX = objetoX.getOrganizacao
    organizacaoY = objetoY.getOrganizacao

    tamanho = 0
    if tamanhox < tamanhoy:
        tamanho = tamanhox
    else:
        tamanho = tamanhoy

    corte = random.randint(0, tamanho)

    novaOrganizacaoX = set()
    novaOrganizacaoY = set()
    for x in organizacaoX:
        i = 0
        if i <= corte:
            novaOrganizacaoX.add(x)
        else:
            novaOrganizacaoY.add(x)

    for x in organizacaoY:
        i = 0
        if i <= corte:
            novaOrganizacaoY.add(x)
        else:
            novaOrganizacaoX.add(x)

    novasOrgs = []
    novasOrgs.append(novaOrganizacaoX)
    novasOrgs.append(novaOrganizacaoY)
    return  novasOrgs

        
def selecao (populacao, fitness):


def algoritimoGenetico(numero_individuos, tamanho_mochila):
    populacao = inicia_populacao(numero_individuos, tamanho_mochila)
    fitness = 0
    for f in populacao:
       fitness += f.getFitness()
    escolidosReproducao = []
    novaPopulacao = []
    for e in escolidosReproducao:
       filhos = reproduzir(e[0], e[1])
       novaPopulacao.append(filhos[0], filhos[1])
    

