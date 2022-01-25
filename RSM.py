from typing import List, Tuple, Set, Dict
import numpy as np
import csv

class Points:
    def __init__(self, a=0, b=0, c=0, d=0, e=0) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def __repr__(self):
        string = '(' + str(self.a) + ', '+ str(self.b)
        if self.c != 0: 
            string += ', '+ str(self.c)
            if self.d != 0: 
                string += ', '+ str(self.d)
                if self.e != 0: 
                    string += ', '+ str(self.e)
        string += ')'
        return string

class Areas:
    """
    Klasa służaca do obrazowania relacji (pole jakie tworzą punkty)
    między dwoma zbiorami: A1 i A2
    """
    def __init__(self, a1, a2, area):
        self.a1 = a1
        self.a2 = a2
        self.area = area

def transfer(points: List[Tuple[int]])->List[object]:
    return [Points(*p) for p in points]

def sorting_points(points: List[object]) -> List[object]:
    """
    Sortowanie punktów w zależności od odległości od punktu (0,0)
    """
    # return sorted(points, key=lambda x: (x[0] ** 2 + x[1] ** 2) ** 0.5)
    return sorted(points, key=lambda x: (x.a ** 2 + x.b ** 2 + x.c ** 2 + x.d ** 2 + x.e ** 2) ** 0.5)

def divide_into_groups(points: object, limit: int) -> Tuple[List[object]]:
    considered_points = points[limit:-limit]
    #ograniczenie zbiorów a następnie podzielenie ich
    candidates_a1 = considered_points[:len(considered_points)//2]
    candidates_a2 = considered_points[len(considered_points)//2:]

    out1 = check_if_points_independant(candidates_a1)
    #usuniecie punktow zaleznych z a1
    a1 = [i for i in candidates_a1 if i not in out1]
    out2 = check_if_points_independant(candidates_a2)
    u = out1
    #usuniecie punktow zaleznych z a2
    a2 = [i for i in candidates_a2 if i not in out2]
    return a1, a2, u

def check_if_points_independant(points: List[object]) -> List[object]:
    # funkcja sprawdzająca  kolejny punkt po punkcie czy kazdy z nich jets punktem niezaleznym w danym podzbiorze
    index = []
    for i in points:
        for j in points:
            if i.a <= j.a and i.b <= j.b and i.c <= j.c and i.d <= j.d and i.e <= j.e and j not in index and i != j:
                index.append(j)
            elif i.a >= j.a and i.b >= j.b and i.c >= j.c and i.d >= j.d and i.e >= j.e and i not in index and i != j:
                index.append(i)
    return index

def diff_area(a1x: object, a2x: object, ux: object)-> int:
    differ = abs(max(a2x, a1x, ux) - min(a2x, a1x, ux))
    return differ if differ != 0 else 1

def field_of_square(A1: List[object], A2: List[object], U: List[object]) -> Dict[object, List[object]]:
    """
    funkcja licząca pola dla wyznaczonych kwadratów w zależności od danego U
    """
    def help_square(A1: List[object], A2: List[object], u: List[object]) -> List[object]:
        """
        Funkcja pomocnicza, liczy pola dla konkretnego, jednego u
        """
        areas = []
        for a1 in A1:
            for a2 in A2:
                area = diff_area(a2.a, a1.a, u.a) * diff_area(a2.b, a1.b, u.b) * diff_area(a2.c, a1.c, u.c) * diff_area(a2.d, a1.d, u.d) * diff_area(a2.e, a1.e, u.e)
                areas.append(Areas(a1, a2, area))
        return areas

    areas = {u: help_square(A1, A2, u) for u in U }
    return areas

def calc_weights(areas: Dict[object, List[object]]) -> Dict[object, List[object]]:

    '''
    Funkcja licząca wagi dla każdego punktu u
    Zwraca słownik {punkt u: lista wag}
    '''

    weights: Dict[object, List[object]] = {}

    for u, fields in areas.items():
        list_of_fields = [A.area for A in fields]
        sum_of_fields = sum(list_of_fields)
        weights_for_one_u = []

        for A in fields:
            weights_for_one_u.append(A.area/sum_of_fields)

        weights[u] = weights_for_one_u
        
    return weights

def calc_distance_coefficients(areas: Dict[object, List[object]]) -> Dict[object, List[object]]:
    
    '''
    Funkcja licząca współczynniki odległości dla każdego punktu u
    Zwraca słownik {punkt u: lista współczynników}
    '''

    distance_coefs: Dict[object, List[object]] = {}

    for u, fields in areas.items():
        distance_coefs_for_one_u = []

        for A in fields:
            u_ = np.array([u.a, u.b, u.c, u.d, u.e])
            a1 = np.array([A.a1.a, A.a1.b, A.a1.c, A.a1.d, A.a1.e])
            a2 = np.array([A.a2.a, A.a2.b, A.a2.c, A.a2.d, A.a2.e])
            d1 = np.linalg.norm(u_ - a1)
            d2 = np.linalg.norm(u_ - a2)
            if d1 > d2:
                distance_coefs_for_one_u.append(d1/(d1 + d2))
            else:
                distance_coefs_for_one_u.append(d2/(d1 + d2))

        distance_coefs[u] = distance_coefs_for_one_u

    return distance_coefs

def calc_score_function(weights: Dict[object, List[object]], distance_coefs: Dict[object, List[object]]) ->  Dict[object, float]:
    
    '''
    Funkcja licząca wartości scoringowe dla każdego punktu u
    Zwraca słownik {punkt u: wartość scoringowa}
    '''

    ranking:  Dict[Tuple[int], float] = {}

    for u in weights:
        w = np.array(weights[u])
        d = np.array(distance_coefs[u])
        ranking[u] = w@d.T

    ranking = {u: value for u, value in sorted(ranking.items(), key=lambda item: item[1],reverse=True)}

    return ranking

def import_csv_file(path):
    hash = {}
    with open(path, newline='') as file:
        reader = list(csv.reader(file))
        for row in reader[1:]:
            name, points = row[1], [float(elem) for elem in row[2:]]
            points = Points(*points)
            hash[points] = name
    return hash

def comparison(ranking, hash):
    named_ranking = {}
    for pos in ranking.keys():
        print(pos)
        print(hash[pos])
        named_ranking[hash[pos]] = pos
    return named_ranking

def show_names(path, ranking):
    hash = import_csv_file(path)
    named = comparison(ranking, hash)
    return named


def main(points, path):
    limit = (len(points)//4) + 1
    points = transfer(points)
    sorting_points(points)
    A1, A2, U = divide_into_groups(points, limit)
    areas = field_of_square(A1, A2, U)

    weights = calc_weights(areas)
    distance_coefs = calc_distance_coefficients(areas)
    ranking = calc_score_function(weights, distance_coefs)

    w = show_names(path, ranking)

    return ranking,A1,A2,w

if __name__ == "__main__":
    main()

