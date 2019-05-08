import random

import numpy as np

from src.Register import Register


class DataSet:
    def __init__(self, data_name, data=None):
        self.data_name = data_name
        if data is None:
            self.data = []
        else:
            self.data = data
        self.stats = None
        self.entropy = None

    def add_register(self, register):
        if register.get_label() is not None:
            self.data.append(register)

    def get_data(self):
        return self.data

    def get_length(self):
        return len(self.data)

    def get_feature_length(self):
        return len(self.data[0].get_features())

    def is_uniform(self):
        labels = self.get_label_statistics()
        if len(labels.keys()) != 1:
            return False
        return True

    def split_on(self, attribute_number, threshold):
        left = []
        right = []
        for elem in self.data:
            if elem.split_left(attribute_number, threshold):
                left.append(elem)
            else:
                right.append(elem)
        left_data = DataSet("left", left)
        right_data = DataSet("right", right)
        return left_data, right_data

    def get_label_statistics(self):
        if self.stats is not None:
            return self.stats
        stats = {}
        for elem in self.data:
            label = elem.get_label()
            if label in stats:
                stats[label] += 1
            else:
                stats[label] = 1
        self.stats = stats
        return self.stats

    def get_entropy(self):
        if self.entropy is not None:
            return self.entropy
        classes = self.get_label_statistics()
        total = len(self.data)
        entropy = 0.0
        for key in classes:
            key_probability = float(classes[key]) / float(total)
            if key_probability != 0:
                entropy = entropy + -1 * key_probability * np.log2(key_probability)
        self.entropy = entropy
        return self.entropy

    def get_best_threshold(self, feature):
        min_entropy = float("inf")
        best_threshold = 0
        for register in self.data:
            threshold = register.get_features()[feature]
            (l_set, r_set) = self.split_on(feature, threshold)
            left_entropy = l_set.get_entropy()
            right_entropy = r_set.get_entropy()
            new_entropy = left_entropy + right_entropy
            if new_entropy <= min_entropy:
                min_entropy = new_entropy
                best_threshold = threshold
        return best_threshold

    def better_threshold(self, feature):
        totalN = self.get_length()
        runningTotal = 0.0

        for samp in self.data:
            runningTotal += samp.get_feature_at_index(feature)

        return float(runningTotal) / totalN

    def get_bag(self, seed=0):
        if seed != 0:
            random.seed(seed)
        bag = []
        for i in range(0, len(self.data)):
            bag.append(random.choice(self.data))
        bag_set = DataSet("bag", bag)
        return bag_set

    def add_register_from_features(self, features, label):
        register = Register(features, label)
        self.add_register(register)

    def get_segments(self, k):
        random_datalist = list(self.data)
        random.shuffle(random_datalist)
        slice_size = len(random_datalist) / k
        data_list = []
        for i in range(0, len(random_datalist), slice_size):
            slice = random_datalist[i:i + slice_size]
            data_list.append(DataSet("slice", slice))
        return data_list

    def normalize_z_score(self):
        n = len(self.data)
        sums = [0 for i in range(len(self.data[0].get_features()))]
        squared_sums = [0 for i in range(len(self.data[0].get_features()))]
        means = [0 for i in range(len(self.data[0].get_features()))]
        std_deviations = [0 for i in range(len(self.data[0].get_features()))]
        for register in self.data:
            features = register.get_features()
            index = 0
            for value in features:
                sums[index] += value
                squared_sums[index] += value * value
                index += 1
        for i in range(len(sums)):
            means[i] = sums[i] / n
            std_deviations[i] = np.sqrt((squared_sums[i] / n) - (means[i] * means[i]))
        for register in self.data:
            register.normalize_z_score(means, std_deviations)

    def normalize_min_max(self, new_mins, new_maxs):
        mins = [float("inf") for i in range(len(self.data[0].get_features()))]
        maxs = [-1 * float("inf") for i in range(len(self.data[0].get_features()))]
        for register in self.data:
            features = register.get_features()
            index = 0
            for value in features:
                if value > maxs[index]:
                    maxs[index] = value
                else:
                    if value < mins[index]:
                        mins[index] = value
                index += 1
        for register in self.data:
            register.normalize_min_max(mins, maxs, new_mins, new_maxs)

    def combine_with_new_data(self, new_data):
        self.data += new_data.get_data()

    def read_data_from_file(self, path):
        input_file = open(path, "r")
        input_file.readline()
        lines = input_file.readlines()
        for line in lines:
            line.strip("\n")
            line_split = line.split(";")
            # 0 - filename, 1 - class, 2 - area, 3 - convex_area, 4 - eccentricity, 5 - filled_area, 6 - perimeter,
            # 7 - solidity
            register = Register(line_split[0],
                                [float(line_split[2]), float(line_split[3]), float(line_split[4]), float(line_split[5]),
                                 float(line_split[6]), float(line_split[7])], line_split[1])
            self.data.append(register)
