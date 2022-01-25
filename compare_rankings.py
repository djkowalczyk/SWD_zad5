import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats


def compare(rank1: np.ndarray, rank2: np.ndarray, type: int):
    if len(rank1) != len(rank2):
        raise ValueError("Rankings are not the same size.")

    if type == 1:
        # suma wartości bezwzględnej różnicy między rankingami
        diff = np.abs(rank1 - rank2)
        distance = np.sum(diff)
        return distance

    elif type == 2:
        # ważona suma wartości bezwzględnej różnicy między rankingami
        rl = len(rank2)
        distance = 0
        for i in range(rl):
            diff = abs(rank1[i] - rank2[i])
            distance = distance + max(rl - rank1[i], rl - rank2[i]) * diff
        return distance

    elif type == 3:
        # Współczynnik korelacji rang Spearmana
        distance, p_value = scipy.stats.spearmanr(rank1, rank2)
        return round(distance, 3)

    elif type == 4:
        # Tau Kendalla
        distance, p_value = scipy.stats.kendalltau(rank1, rank2)
        return round(distance, 3)

    else:
        raise ValueError("Wrong type")


def graph(matrix: np.ndarray, methods: list):
    G = nx.Graph()
    s = np.shape(matrix)
    list_G = []
    for i in range(s[0]):
        for j in range(s[0]):
            if j > i:
                list_G.append((i + 1, j + 1, matrix[i, j]))

    labeldict = {i+1: method for i, method in enumerate(methods)}

    G.add_weighted_edges_from(list_G)
    edge_weight = nx.get_edge_attributes(G, 'weight')
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos=pos, width=2, node_size=1000, labels=labeldict, with_labels=True, font_size=10, font_color="red")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_weight)
    plt.title("Porównanie rankingów")
    plt.show()


def drawDistances(rank: np.ndarray, type: int, methods: list):
    sr = np.shape(rank)
    D = np.zeros((sr[0], sr[0]))

    for i in range(sr[0]):
        for j in range(sr[0]):
            if i == j:
                D[i, j] = 0
            else:
                D[i, j] = compare(rank[i], rank[j], type)

    graph(matrix=D, methods=methods)


def main(rankings, type, methods):
    rank = np.array(rankings)
    drawDistances(rank, type, methods)


if __name__ == '__main__':
    r1 = [1, 3, 4, 2]
    r2 = [2, 4, 1, 3]
    r3 = [2, 2, 1, 4]
    rank = np.array([r1, r2, r3])
    methods = ["RSM", "TOPSIS", "UTA"]
    drawDistances(rank, 3, methods)
