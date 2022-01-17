"""Główny moduł zawierający implementację metody TOPSIS, sposoby na prezentację wyników końcowych(rankingu) oraz
testy metody TOPSIS dla dwóch zestawów danych - smartfonów oraz laptopów."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, NamedTuple, Dict, Tuple, Union


class Solution(NamedTuple):
    """Struktura rozwiązania problemu decyzyjnego."""

    n_dimensions: int
    alternatives: np.ndarray
    classes: np.ndarray
    decision_matrix: np.ndarray
    scaled_decision_matrix: np.ndarray
    ideal_vector: np.ndarray
    anti_ideal_vector: np.ndarray
    distances: np.ndarray
    ranking: np.ndarray

    def to_excel(self, alternatives_names: list):

        data = {'Wynik': self.ranking,
                'Nazwa alternatywy': alternatives_names}

        df = pd.DataFrame(data)

        df.to_excel('ranking.xlsx')

    def get_dict_ranking(self) -> Dict[Tuple[Union[int, float]], float]:
        dict_ranking = {}
        for i, score in enumerate(self.ranking):
            features = tuple(self.alternatives[i, 2:])
            dict_ranking[features] = score
        return dict_ranking


def display_solution(s: Solution, axis_labels: List[str], title: str, alternatives_names: List[str] = None) -> None:
    """Przedstawia rozwiązanie w przystępnej dla człowieka formie. W zależności od wymiaru problemu wyświetlany jest
    wykres 2D albo 3D. Prezentacja problemów o wyższej wymiarowości nie jest obecnie wspierana.

    Args:
        s: rozwiązanie uzyskane metodą topsis w postaci nazwanej krotki dla zwiększenia spójności kodu.
        axis_labels: lista podpisów poszczególnych osi w kolejności x, y, (z).
        title: tytuł końcowego wykresu.
        alternatives_names: lista nazw alternatyw do wyświetlenia w rankingu

    Returns:
        Funkcja wypisuje istotne informacje na ekranie oraz jeżeli wymiar problemu jest nie większy niż 3 to
        wyświetlany zostaje wykres przedstawiający położenie ocen rankingowych alternatyw.

     """

    # wypisanie wyników
    print("-" * 40)
    print("Macierz decyzyjna")
    for row in s.decision_matrix:
        print(row)
    print("-" * 40)
    print("Macierz skalowana")
    for row in s.scaled_decision_matrix:
        print(row)
    print("-" * 40)
    print("Wektor idealny")
    print(s.ideal_vector)
    print("-" * 40)
    print("Wektor antyidealny")
    print(s.anti_ideal_vector)
    print("-" * 40)
    print("Macierz odległości")
    for row in s.distances:
        print(row)
    print("-" * 40)
    print("Ranking")
    if alternatives_names is None:
        for i, value in enumerate(s.ranking):
            print(f"{i + 1} : {value}")
    else:
        print("Wynik", "Nazwa alternatywy", sep='\t\t\t\t')
        for score, name in sorted([(score, name) for score, name in zip(s.ranking.tolist(), alternatives_names)],
                                  key=lambda x: x[0]):
            print(score, name, sep='\t')
        s.to_excel(alternatives_names=alternatives_names)
    print("-" * 40)

    if s.n_dimensions == 2:
        # wykres 2D
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes()
        ax.scatter(
            s.decision_matrix[:, 0],
            s.decision_matrix[:, 1],
            marker="+",
            s=40,
            color="blue",
        )
        ax.scatter(s.classes[:, 1], s.classes[:, 2], marker="o", s=40, color="red")
        ax.set_title(title)
        xlabel, ylabel = axis_labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.show()

    elif s.n_dimensions == 3:
        # wykres 3D
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            s.decision_matrix[:, 0],
            s.decision_matrix[:, 1],
            s.decision_matrix[:, 2],
            marker="+",
            s=40,
            color="blue",
        )
        ax.scatter3D(
            s.classes[:, 1],
            s.classes[:, 2],
            s.classes[:, 3],
            marker="o",
            s=40,
            color="red",
        )
        ax.set_title(title)
        xlabel, ylabel, zlabel = axis_labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        # ustawienie odpowiedniej skali osi y (problem z domyślnymi ustawieniami w matplotlibie)
        ax.set_ylim(np.min(s.classes[:, 2]), np.max(s.classes[:, 2]))
        ax.legend(['alternatywy', 'punkt idealny i antyidealny'], bbox_to_anchor=(1.1, 1.0))
        fig.show()

    else:
        print(f"Obecnie nie wspieramy wizualizacji {s.n_dimensions}-wymiarowych problemów.")


def euclidean_metric(n_alt: int, n_criteria: int, scaled_decision_matrix: np.ndarray, ideal_vector: np.ndarray, anti_ideal_vector: np.ndarray) -> np.ndarray:
    """Funkcja wyznaczania odległości w metryce euklidesowej"""
    distances = np.zeros((n_alt, 2))
    for i in range(n_alt):
        sum_ideal = 0
        sum_anti_ideal = 0
        for j in range(1, n_criteria):
            sum_ideal += np.square(scaled_decision_matrix[i, j] - ideal_vector[j])
            sum_anti_ideal += np.square(scaled_decision_matrix[i, j] - anti_ideal_vector[j])

        distances[i, 0] = np.sqrt(sum_ideal)
        distances[i, 1] = np.sqrt(sum_anti_ideal)
    return distances


def chebyshev_metric(n_alt: int, n_criteria: int, scaled_decision_matrix: np.ndarray, ideal_vector: np.ndarray, anti_ideal_vector: np.ndarray) -> np.ndarray:
    """Funkcja wyznaczania odległości w metryce czebyszewa"""
    distances = np.zeros((n_alt, 2))
    for i in range(n_alt):
        ideal = np.empty(n_criteria)
        anti_ideal = np.empty(n_criteria)
        for j in range(1, n_criteria):
            ideal[j] = abs(scaled_decision_matrix[i, j] - ideal_vector[j])
            anti_ideal[j] = abs(scaled_decision_matrix[i, j] - anti_ideal_vector[j])
        sum_ideal = np.max(ideal)
        sum_anti_ideal = np.max(anti_ideal)

        distances[i, 0] = sum_ideal
        distances[i, 1] = sum_anti_ideal
    return distances


def manhattan_metric(n_alt: int, n_criteria: int, scaled_decision_matrix: np.ndarray, ideal_vector: np.ndarray, anti_ideal_vector: np.ndarray) -> np.ndarray:
    """Funkcja wyznaczania odległości w metryce taksówkowej"""
    distances = np.zeros((n_alt, 2))
    for i in range(n_alt):
        sum_ideal = 0
        sum_anti_ideal = 0
        for j in range(1, n_criteria):
            sum_ideal += abs(scaled_decision_matrix[i, j] - ideal_vector[j])
            sum_anti_ideal += abs(scaled_decision_matrix[i, j] - anti_ideal_vector[j])

        distances[i, 0] = np.sqrt(sum_ideal)
        distances[i, 1] = np.sqrt(sum_anti_ideal)
    return distances


def topsis(alternatives: np.ndarray, classes: np.ndarray, weights: np.ndarray, criteria_types: List[str],
           metric: callable) -> Solution:
    """Metoda TOPSIS(Technique for Order of Preference by Similarity to Ideal Solution).

    Args:
        alternatives: macierz alternatyw o wymiarach (n_alt, n_criteria)
        classes: macierz klas o wymiarach (n_classes, n_criteria)
        weights: wektor wag o długości n_criteria
        criteria_types: lista rodzai kryteriów (maksymalizowanych albo minimalizowanych)
        metric: funkcja wyznaczenia odległości w danej metryce

    Returns:
        s: rozwiązanie problemu decyzyjnego

    """
    # określenie wielkości problemu
    n_alt, n_criteria = alternatives.shape
    n_criteria -= 2

    # określenie ilości klas
    n_classes, n_dimensions = classes.shape
    n_dimensions -= 1

    # sprawdzenie punktów alternatyw
    alt_points = np.zeros(n_alt)
    for i in range(n_alt):
        for j in range(2, n_criteria + 2):
            if (criteria_types[j - 2] == "max" and classes[0, j - 1] <= alternatives[i, j] <= classes[1, j - 1])\
                    or (criteria_types[j - 2] == "min" and classes[0, j - 1] >= alternatives[i, j] >= classes[1, j - 1]):
                alt_points[i] = i + 1
            else:
                alt_points[i] = 0
                break

    n_alt_temp = 0
    for i in range(n_alt):
        if alt_points[i] == 0:
            n_alt_temp += 1

    # uzupełnienie macierzy decyzjnej
    id_ = 0
    decision_matrix = np.zeros((n_alt_temp, n_criteria))
    for i in range(n_alt):
        if alt_points[i] == 0:
            for j in range(2, n_criteria + 2):
                decision_matrix[id_, j - 2] = alternatives[i, j]
        id_ += 1
    n_alt = n_alt_temp

    # proces skalowania macierzy decyzyjnej
    scaled_decision_matrix = np.zeros((n_alt, n_criteria))
    for i in range(n_alt):
        for j in range(n_criteria):
            scaled_decision_matrix[i, j] = \
                (decision_matrix[i, j] * weights[j]) / np.sqrt(np.sum(np.square(decision_matrix[:, j])))

    # wyznaczenie wektora idealnego i antyidealnego
    ideal_vector = np.array([
        np.min(scaled_decision_matrix[:, i]) if criteria_types[i] == 'min' else np.max(scaled_decision_matrix[:, i])
        for i in range(n_dimensions)
    ])

    anti_ideal_vector = np.array([
        np.max(scaled_decision_matrix[:, i]) if criteria_types[i] == 'min' else np.min(scaled_decision_matrix[:, i])
        for i in range(n_dimensions)
    ])

    # wyznaczenie odległości
    distances = metric(n_alt, n_criteria, scaled_decision_matrix, ideal_vector, anti_ideal_vector)

    # uszeregowanie obiektów
    ranking = np.zeros(n_alt)
    for i in range(n_alt):
        ranking[i] = distances[i, 1] / (distances[i, 0] + distances[i, 1])

    return Solution(
        n_dimensions,
        alternatives,
        classes,
        decision_matrix,
        scaled_decision_matrix,
        ideal_vector,
        anti_ideal_vector,
        distances,
        ranking,

    )


def test_smartphones():
    """Test metody TOPSIS dla zestawu smartfonów."""
    print()
    print('Test dla smartfonów')
    print()
    A = [
        [1, 1, 700, 2, 6],
        [1, 1, 250, 2, 4],
        [1, 1, 1200, 6, 9],
        [1, 1, 880, 4, 9],
        [1, 1, 450, 3, 8],
        [1, 1, 1500, 2, 3],
    ]

    K = [[1, 1500, 1, 3], [1, 200, 7, 10]]
    W = [0.45, 0.3, 0.25]
    C = ['min', 'max', 'max']

    s = topsis(alternatives=np.array(A), classes=np.array(K), weights=np.array(W), criteria_types=C, metric=euclidean_metric)
    display_solution(s, axis_labels=["Cena", "RAM", "Wygląd"], title="Położenie ocen rankingowych alternatyw")


def test_laptops():
    """Test metody TOPSIS dla zestawu laptopów."""
    from laptop_input_handler import import_laptop_data
    print()
    print('Test dla laptopów')
    print()
    laptop_names, A, K, C = import_laptop_data()

    # poprawka na offsety
    A = [[1, 1] + row for row in A]

    # TODO: K wystarczy odpowiednio zmodyfikować aby zamiast ideal_vector i anti_ideal_vector
    # był najlepszy punkt z zbioru docelowanego i nadir ze zbioru status quo
    K = [[1] + row for row in K]

    # wektor wag może być w przyszłości podawany przez użytkownika
    W = [0.1] * 10
    s = topsis(alternatives=np.array(A), classes=np.array(K), weights=np.array(W), criteria_types=C, metric=euclidean_metric)
    display_solution(s, axis_labels=[], title="", alternatives_names=laptop_names)


def test_batteries():
    """Test metody TOPSIS dla zestawu baterii."""
    from battery_input_handler import import_battery_data

    print()
    print('Test dla baterii')
    print()
    battery_names, A, K, C = import_battery_data()

    # poprawka na offsety
    A = [[1, 1] + row for row in A]

    # TODO: K wystarczy odpowiednio zmodyfikować aby zamiast ideal_vector i anti_ideal_vector
    # był najlepszy punkt z zbioru docelowanego i nadir ze zbioru status quo
    K = [[1] + row for row in K]

    # wektor wag może być w przyszłości podawany przez użytkownika
    W = [0.33, 0.33, 0.33]
    s = topsis(alternatives=np.array(A), classes=np.array(K), weights=np.array(W), criteria_types=C, metric=manhattan_metric)
    display_solution(s, axis_labels=["Cena", "V", "mAh"], title="Położenie ocen rankingowych alternatyw", alternatives_names=battery_names)


if __name__ == "__main__":
    #test_laptops()
    test_batteries()
    #test_smartphones()

