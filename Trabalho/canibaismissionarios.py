def son2str(s):
    return ' '.join([str(v) for v in s])


def isGoal(s):
    goal = False
    if (s[0] == 0) and (s[1] == 0) and (s[2] == 0):
        goal = True
    return goal


def generateSons(s):
    listOfSons = []

    if s[2] == 1:

        if (s[0] > 0) and (s[0] - s[1] > 0):
            # -1 misiionario
            state = [s[0] - 1, s[1], 0]
            listOfSons.append(state)

        if s[1] > 0:
            # -1 canibal
            state = [s[0], s[1] - 1, 0]
            listOfSons.append(state)

    else:
        # canoa na margem esquera
        state = [s[0], s[1], 1]
        listOfSons.append(state)

    return listOfSons


def bfs(start):
    candidates = [start]
    fathers = dict()
    visited = [start]

    while len(candidates) > 0:
        father = candidates[0]
        print("Lista de candidatos: ", candidates)
        del candidates[0]
        print("Visitado: ", father)
        if isGoal(father):
            res = []
            node = father
            while node != start:
                res.append(node)
                node = fathers[son2str(node)]
            res.append(start)
            res.reverse()
            print("Solucao encontrada: ", res)

        for son in generateSons(father):
            if son not in visited:
                print("Enfileirado: ", son, father)
                visited.append(son)
                fathers[son2str(son)] = father
                candidates.append(son)


if __name__ == '__main__':
    bfs([3, 3, 1])
