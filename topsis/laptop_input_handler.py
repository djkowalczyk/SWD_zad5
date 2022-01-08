"""Moduł importujący dane z pliku csv - w przyszłości lepszym rozwiązaniem na pewno będzie wymuszenie odpowiedniego
formatowania danych wejściowych na użytkowniku, bądź odpowiednie zautomatyzowanie procesu wprowadzania danych."""


def import_laptop_data(filepath: str = 'data/Laptopy.csv'):
    """Importuje dane dotyczące zestawu laptopów.

    Args:
        filepath: ścieżka do pliku csv zawierającego dane o laptopach.

    Returns:
        laptop_names: lista nazw laptopów.
        alternatives: lista parametrów każdego z laptopów.
        classes: lista (min, max) wartości z każdego parametru laptopów.
        criteria_types: lista określająca czy dany paramtery ma być minimalizowany czy też maksymalizowany.

    """
    with open(filepath, 'r') as f:
        categories = f.readline()[3:-1].split(";")
        laptop_data = [line[:-1].split(";") for line in f.readlines()]

    laptop_names = [f'{row[0]} {row[1]}' for row in laptop_data]
    cores = [int(row[3]) for row in laptop_data]
    clock_speeds = [float(row[4].replace(",", ".")) for row in laptop_data]
    resolutions = [int(row[5]) * int(row[6]) for row in laptop_data]
    battery_work_time = [float(row[9].strip('h')) for row in laptop_data]
    ssd = [int(row[7]) for row in laptop_data]
    ram = [int(row[8]) for row in laptop_data]
    prices = [int(row[10]) for row in laptop_data]
    warranties = [int(row[11]) for row in laptop_data]
    gpu_ratings = [float(row[13].replace(",", ".")) for row in laptop_data]
    screen_sizes = [float(row[14].replace(",", ".")) for row in laptop_data]

    categories = [cores,
                  clock_speeds,
                  resolutions,
                  battery_work_time,
                  ssd,
                  ram,
                  prices,
                  warranties,
                  gpu_ratings,
                  screen_sizes]

    alternatives = [[c[i] for c in categories] for i in range(len(cores))]

    # w przyszłości ta lista powinna być definiowana przez użytkownika
    criteria_types = ['max', 'max', 'max', 'max', 'max', 'max', 'min', 'max', 'max', 'max']

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

    return laptop_names, alternatives, classes, criteria_types
