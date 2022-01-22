# Metoda UTASTAR
# Utworzone przez Hubert Czader, Filip Gąciarz, Julia Grubska, Rafał Kośla,
# Dorota Kowalczyk, Zofia Lenarczyk, Adam Stacherski, Dominik Żurek


import csv


def import_csv_file(path):
    with open(path, newline='') as file:
        reader = list(csv.reader(file))
        data = []
        labels = reader[0]
        size = len(labels) - 2
        for row in reader[1:]:
            name, points = row[1], [float(elem) for elem in row[2:]]
            data.append(CraftBeer(name=name, points=points))
        return Utastar(data, labels, size)


class CraftBeer(object):
    def __init__(self, name, points):
        self.name = name
        self.points = points
        self.u_values = []
        self.utastar_value = 0

    # Opisy funkcji lower than i greater than są na odwrót, gdyż lepszy współczynnik jest ten który jest większy,
    # a sortując końcowo chcemy uzyskać listę od najbardziej opłacalnego piwa do najmniej
    def __lt__(self, other):
        return self.utastar_value > other.utastar_value

    def __gt__(self, other):
        return self.utastar_value < other.utastar_value


class Utastar:
    def __init__(self, data_base, labels, size):
        self.data_base = data_base
        self.labels = labels
        self.rows = len(data_base)
        self.size = size
        self.parts = None
        self.function_values = None
        self.function_values_in_intervals = None

    def insert(self, craft_beer):
        if craft_beer is None:
            raise Exception("Nie znaleziono elementu!")
        else:
            self.data_base.append()
            self.rows += 1

    def find_min(self):
        if self.data_base is None:
            raise Exception("Baza danych nie istnieje!")
        else:
            minimum = []
            for i in range(self.size):
                act_min = self.data_base[0].points[i]
                for craft_beer in self.data_base[1:]:
                    if craft_beer.points[i] < act_min:
                        act_min = craft_beer.points[i]
                minimum.append(act_min)
            return minimum

    def find_max(self):
        if self.data_base is None:
            raise Exception("Baza danych nie istnieje!")
        else:
            maximum = []
            for i in range(self.size):
                act_min = self.data_base[0].points[i]
                for craft_beer in self.data_base[1:]:
                    if craft_beer.points[i] > act_min:
                        act_min = craft_beer.points[i]
                maximum.append(act_min)
            return maximum

    def create_ideal_point(self):
        minimum = self.find_min()
        return minimum

    def create_anti_ideal_point(self):
        maximum = self.find_max()
        return maximum

    def sorting(self):
        if self.data_base is None:
            raise Exception("Baza danych nie istnieje!")
        else:
            self.data_base.sort()

    def divide_into_parts(self, num_of_parts: list = None):
        if num_of_parts is None:
            # num_of_parts = [8, 6, 3, 7, 4]
            num_of_parts = [8, 8, 8, 8, 8]
            num_of_parts = num_of_parts[:self.size]
        if len(num_of_parts) != self.size:
            raise Exception("Podano nieprawidłową liczbę współczynników!")
        else:
            if self.parts is None:
                minimum = self.find_min()
                maximum = self.find_max()
                sum_components = [(maximum[i] - minimum[i]) / (num_of_parts[i] - 1) for i in range(self.size)]
                self.parts = []
                for i in range(self.size):
                    self.parts.append([minimum[i] + j * sum_components[i] for j in range(num_of_parts[i] - 1)] +
                                      [maximum[i]])
            else:
                raise Exception("Podział na części został już wykonany!")

    def print_parts(self):
        print("\nPodział na części:")
        for i in range(self.size):
            print("{}: ".format(self.labels[i + 2]), self.parts[i])

    def create_usability_function_values(self, weights: list = None):
        if self.function_values is None:
            if weights is None:
                # weights = [[0.2, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03, 0], [0.15, 0.13, 0.1, 0.06, 0.03, 0],
                #            [0.25, 0.15, 0], [0.1, 0.09, 0.075, 0.045, 0.03, 0.01, 0], [0.3, 0.22, 0.1, 0]]
                weights = [[0.2, 0.18, 0.14, 0.11, 0.09, 0.05, 0.03, 0], [0.2, 0.18, 0.14, 0.11, 0.09, 0.05, 0.03, 0],
                           [0.2, 0.18, 0.14, 0.11, 0.09, 0.05, 0.03, 0], [0.2, 0.18, 0.14, 0.11, 0.09, 0.05, 0.03, 0],
                           [0.2, 0.18, 0.14, 0.11, 0.09, 0.05, 0.03, 0]]
                weights = [[i * 5/self.size for i in elem] for elem in weights]
                weights = weights[:self.size]
            sum = 0
            for i in range(self.size):
                if len(self.parts[i]) != len(weights[i]):
                    raise Exception("Podano nieprawidłową liczbę wag!")
                sum += weights[i][0]
            if sum != 1:
                raise Exception("Ogólna funkcja użyteczności, czyli suma wartości funkcji użyteczności na "
                                "poszczególnych kryteriach powinna sumować się do 1 dla punktu idealnego")
            self.function_values = []
            for i in range(self.size):
                num_of_parts = len(self.parts[i])
                value = {self.parts[i][j]: weights[i][j] for j in range(num_of_parts)}
                self.function_values.append(value)
        else:
            raise Exception("Wartości funkcji użyteczności już istnieją!")

    def create_functions(self):
        if self.function_values is None:
            raise Exception("Wartości funkcji użyteczności nie istnieją!")
        else:
            if self.function_values_in_intervals is None:
                self.function_values_in_intervals = []
                for i in range(self.size):
                    print("\nFunkcja użyteczności u{} określona jest wzorem: y = ax + b".format(i + 1))
                    key_list = []
                    value_list = []
                    for key, value in self.function_values[i].items():
                        key_list.append(key)
                        value_list.append(value)
                    values_for_interval = {}
                    counter = 1
                    for j in range(len(self.function_values[i]) - 1):
                        a = (value_list[j] - value_list[j + 1]) / (key_list[j] - key_list[j + 1])
                        b = value_list[j] - a * key_list[j]
                        print("W przedziale {}:  a = {}, b = {}".format(counter, a, b))
                        values_for_interval[(key_list[j], key_list[j + 1])] = (a, b)
                        counter += 1
                    self.function_values_in_intervals.append(values_for_interval)
            else:
                raise Exception("Wartości funkcji są już określone na przedziałach!")

    def create_solution_table(self):
        if self.data_base is None:
            raise Exception("Baza danych nie istnieje!")
        elif self.function_values_in_intervals is None:
            raise Exception("Nie można wygenerować rozwiązania, funkcje nie zostały utworzone!")
        else:
            for craft_beer in self.data_base:
                if not craft_beer.u_values:
                    for i in range(self.size):
                        for key, value in self.function_values_in_intervals[i].items():
                            if key[0] <= craft_beer.points[i] <= key[1]:
                                craft_beer.u_values.append(value[0] * craft_beer.points[i] + value[1])
                                break
                    for u in craft_beer.u_values:
                        craft_beer.utastar_value += u
                else:
                    continue
            self.sorting()

    def return_solution(self):
        solution = {}
        for craft_beer in self.data_base:
            key = [craft_beer.name] + craft_beer.points
            key = tuple(key)
            value = craft_beer.utastar_value
            solution[key] = value
        return solution


def main(path):
    data_base = import_csv_file(path)
    v_ideal = data_base.create_ideal_point()
    v_anti = data_base.create_anti_ideal_point()
    data_base.divide_into_parts()
    data_base.print_parts()
    data_base.create_usability_function_values()
    data_base.create_functions()
    data_base.create_solution_table()
    return data_base.return_solution()






