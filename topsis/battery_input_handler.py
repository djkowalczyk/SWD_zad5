def minmax_scale(data):
    x_max = max(data)
    x_min = min(data)
    return [(x-x_min)/(x_max-x_min) for x in data]


def import_battery_data(filepath: str = 'data/Baza.csv'):
    with open(filepath, 'r') as file:
        category_names = file.readline()[3:-1].split(';')
        data = [line.split(";") for line in file.readlines()]

    x = [float(row[0]) for row in data]
    y = [float(row[1]) for row in data]
    z = [int(row[2]) for row in data]
    
    # x = minmax_scale(x)
    # y = minmax_scale(y)
    # z = minmax_scale(z)

    categories = [x, y, z]

    alternatives = [[c[i] for c in categories] for i in range(len(data))]

    criteria_types = ['min', 'max', 'max']

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

    battery_names = [f'Bateria {i+1}' for i in range(len(data))]

    return battery_names, alternatives, classes, criteria_types


if __name__ == '__main__':
    import_battery_data()
