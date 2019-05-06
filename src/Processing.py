from src.Data import string2data, ECCENTRICITY_COLUMN, AREA_COLUMN, CONVEX_AREA_COLUMN, FILLED_AREA_COLUMN, \
    PERIMETER_COLUMN, SOLIDITY_COLUMN


def get_all_data(path):
    input_file = open(path, "r")
    input_file.readline()
    lines = input_file.readlines()
    data_list = []
    for line in lines:
        data = string2data(line)
        data_list.append(data)
    return data_list


def min_max_normalize(data_list, column):
    values = []
    for data in data_list:
        if column == AREA_COLUMN:
            values.append(data.area)
        if column == CONVEX_AREA_COLUMN:
            values.append(data.convex_area)
        if column == ECCENTRICITY_COLUMN:
            values.append(data.eccentricity)
        if column == FILLED_AREA_COLUMN:
            values.append(data.filled_area)
        if column == PERIMETER_COLUMN:
            values.append(data.perimeter)
        if column == SOLIDITY_COLUMN:
            values.append(data.solidity)
    min_value = min(values)
    max_value = max(values)
    for data in data_list:
        if column == AREA_COLUMN:
            data.area = (data.area - min_value) / (max_value - min_value)
        if column == CONVEX_AREA_COLUMN:
            data.convex_area = (data.convex_area - min_value) / (max_value - min_value)
        if column == ECCENTRICITY_COLUMN:
            data.eccentricity = (data.eccentricity - min_value) / (max_value - min_value)
        if column == FILLED_AREA_COLUMN:
            data.filled_area = (data.filled_area - min_value) / (max_value - min_value)
        if column == PERIMETER_COLUMN:
            data.perimeter = (data.perimeter - min_value) / (max_value - min_value)
        if column == SOLIDITY_COLUMN:
            data.solidity = (data.solidity - min_value) / (max_value - min_value)
