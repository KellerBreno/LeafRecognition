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
        labels = self.get_dataset_statistics()
        if len(labels.keys()) != 1:
            return False
        return True

    def split_on(self, feature_index, threshold):
        left = []
        right = []
        for elem in self.data:
            if elem.is_less_than(feature_index, threshold):
                left.append(elem)
            else:
                right.append(elem)
        left_data = DataSet("left", left)
        right_data = DataSet("right", right)
        return left_data, right_data

    def get_dataset_statistics(self):
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
        classes = self.get_dataset_statistics()
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

    def get_threshold(self, feature):
        dataset_length = self.get_length()
        total = 0.0

        for register in self.data:
            total += register.get_feature_at_index(feature)

        return float(total) / dataset_length

    def get_random_subset(self, seed=0):
        if seed != 0:
            random.seed(seed)
        subset = []
        for i in range(0, len(self.data)):
            # Amostragem com repetição
            subset.append(random.choice(self.data))
        subset = DataSet("subset", subset)
        return subset

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

    def read_data_from_file(self, path):
        input_file = open(path, "r")
        input_file.readline()
        lines = input_file.readlines()
        for line in lines:
            line.strip("\n")
            line_split = line.split(";")
            # 0 - filename, 1 - class, 2 - area, 3 - convex_area, 4 - eccentricity, 5 - filled_area, 6 - perimeter,
            # 7 - solidity, 8 - extent, 9 - orientation
            register = Register(line_split[0],
                                [float(line_split[2]), float(line_split[3]), float(line_split[4]), float(line_split[5]),
                                 float(line_split[6]), float(line_split[7]), float(line_split[8]),
                                 float(line_split[9])], line_split[1])
            # register = Register(line_split[0],
            #                     [float(line_split[2]), float(line_split[3]), float(line_split[4]),
            #                     float(line_split[5]), float(line_split[6]), float(line_split[7])], line_split[1])
            self.data.append(register)

    def export_to_file(self, path):
        output_file = open(path, "w+")
        output_file.write(
            "filename,class,area,convex_area,eccentricity,filled_area,perimeter,solidity,extent,orientation\n")
        for register in self.data:
            output_file.write(
                register.get_filename() + "," + register.get_label() + "," + str(
                    register.get_feature_at_index(0)) + "," + str(register.get_feature_at_index(1)) + "," + str(
                    register.get_feature_at_index(2)) + "," + str(register.get_feature_at_index(3)) + "," + str(
                    register.get_feature_at_index(4)) + "," + str(register.get_feature_at_index(5)) + "," + str(
                    register.get_feature_at_index(6)) + "," + str(register.get_feature_at_index(7)) + "\n")
        output_file.close()

    def get_labels(self):
        labels = []
        for i in range(0, len(self.data)):
            labels.append(self.data[i].get_label())
        return labels

    def get_features(self):
        features = []
        for i in range(0, len(self.data)):
            features.append(self.data[i].get_features())
        return features

    def get_filenames(self):
        ids = []
        for i in range(0, len(self.data)):
            ids.append(self.data[i].get_filename())
        return ids

    def get_balanced_dataset(self):
        list_label = []
        for label in self.get_labels():
            aux = True
            for l in list_label:
                if l == label:
                    aux = False
                    break
            if aux:
                list_label.append(label)
        i = len(list_label)
        data_list = [[] for j in range(i)]
        for j in range(0, i):
            data_list[j] = []
        for d in self.data:
            aux = 0
            for label in list_label:
                if d.get_label() == label:
                    data_list[aux].append(d)
                    break
                aux = aux + 1
        min = float("inf")
        for j in range(0, i):
            if len(data_list[j]) < min:
                min = len(data_list[j])
        for j in range(0, i):
            exclude = len(data_list[j]) - min
            for k in range(0, exclude):
                sort = random.randrange(0, len(data_list[j]))
                del data_list[j][sort]
        new_data = []
        for j in range(0, i):
            new_data = new_data + data_list[j]
        return DataSet("balanced", new_data)
