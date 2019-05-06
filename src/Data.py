FILENAME_COLUMN = 0
CLASSNAME_COLUMN = 1
AREA_COLUMN = 2
CONVEX_AREA_COLUMN = 3
ECCENTRICITY_COLUMN = 4
FILLED_AREA_COLUMN = 5
PERIMETER_COLUMN = 6
SOLIDITY_COLUMN = 7


def string2data(line):
    line.strip("\n")
    line_split = line.split(";")
    return Data(filename=line_split[0], classname=line_split[1], area=float(line_split[2]),
                convex_area=float(line_split[3]), eccentricity=float(line_split[4]), filled_area=float(line_split[5]),
                perimeter=float(line_split[6]), solidity=float(line_split[7]))


class Data:
    def __init__(self, filename, classname, area, convex_area, eccentricity, filled_area, perimeter, solidity):
        self.filename = filename
        self.classname = classname
        self.area = area
        self.convex_area = convex_area
        self.eccentricity = eccentricity
        self.filled_area = filled_area
        self.perimeter = perimeter
        self.solidity = solidity
