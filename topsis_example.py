import numpy as np
from topsis import topsis, manhattan_metric, Solution


def prepare_dataset_for_topsis(filepath: str):
    with open(filepath, 'r') as file:
        category_names = file.readline()[1:].split(',')
        data = [line.split(",") for line in file.readlines()]

    names = [row[1] for row in data]
    percents = [float(row[3]) for row in data]
    ibus = [float(row[3]) for row in data]
    volumes = [int(row[4]) for row in data]
    commonness = [float(row[5]) for row in data]
    prices = [float(row[6]) for row in data]

    categories = [percents, ibus, volumes, commonness, prices]

    alternatives = [[c[i] for c in categories] for i in range(len(data))]

    criteria_types = ['min', 'min', 'min', 'min', 'min']

    ideal_point = []
    anti_ideal_point = []
    for criteria_type, category in zip(criteria_types, categories):
        if criteria_type == 'min':
            ideal_point.append(min(category))
            anti_ideal_point.append(max(category))
        else:
            ideal_point.append(max(category))
            anti_ideal_point.append(min(category))

    classes = [ideal_point,
               anti_ideal_point]

    return names, category_names, alternatives, classes, criteria_types


if __name__ == '__main__':
    beer_data_path = 'datasets/piwa_kraftowe.csv'

    names, category_names, alternatives, classes, criteria_types = prepare_dataset_for_topsis(beer_data_path)

    # poprawka na offsety
    A = [[1, 1] + row for row in alternatives]

    # najlepszy punkt z zbioru docelowanego i nadir ze zbioru status quo
    K = [[1] + row for row in classes]

    # wektor wag
    W = [1 / len(criteria_types)] * len(criteria_types)

    s: Solution = topsis(alternatives=np.array(A), classes=np.array(K), weights=np.array(W),
                         criteria_types=criteria_types, metric=manhattan_metric)

    # metoda elegancko zwraca słownik (współrzędne punktu) : wartość scoringowa
    for key, value in s.get_dict_ranking().items():
        print(key, value)
