def filhosToString(s):
    return ' '.join([str(v) for v in s])


def objetivo(s):
    goal = False
    if (s[0] == 0) and (s[1] == 0) and (s[2] == 0):
        goal = True
    return goal


def geraFilhos(s):
    listaDeFilhos = []

    if s[2] == 1:

        if (s[0] > 0) and (s[0] - s[1] > 0):
            # -1 misiionario
            estado = [s[0] - 1, s[1], 0]
            listaDeFilhos.append(estado)

        if s[1] > 0:
            # -1 canibal
            estado = [s[0], s[1] - 1, 0]
            listaDeFilhos.append(estado)

    else:
        # canoa na margem esquera
        estado = [s[0], s[1], 1]
        listaDeFilhos.append(estado)

    return listaDeFilhos


def busca(inicio):
    candidatos = [inicio]
    pais = dict()
    visitados = [inicio]

    while len(candidatos) > 0:
        pai = candidatos[0]
        print("Lista de candidatos: ", candidatos)
        del candidatos[0]
        print("Visitado: ", pai)
        if objetivo(pai):
            res = []
            node = pai
            while node != inicio:
                res.append(node)
                node = pais[filhosToString(node)]
            res.append(inicio)
            res.reverse()
            print("Solucao encontrada: ", res)

        for son in geraFilhos(pai):
            if son not in visitados:
                print("Enfileirado: ", son, pai)
                visitados.append(son)
                pais[filhosToString(son)] = pai
                candidatos.append(son)


if __name__ == '__main__':
    busca([3, 3, 1])
