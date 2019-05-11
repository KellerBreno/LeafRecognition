import numpy as np


class Register:
    def __init__(self, filename=None, features=None, label=None):
        self.filename = filename
        self.features = features
        self.label = label

    def set_features(self, features):
        if (type(features) != np.ndarray) or (len(features.shape) != 1):
            raise ValueError("The feature vector is not a single dimensional numpy array!")
        self.features = features

    def normalize_z_score(self, means, std_deviations):
        index = 0
        for index in range(len(self.features)):
            z = (self.features[index] - means[index]) / std_deviations[index]
            self.features[index] = z
            index += 1

    def normalize_min_max(self, mins, maxs, new_mins, new_maxs):
        index = 0
        for index in range(len(self.features)):
            v = ((self.features[index] - mins[index]) / (maxs[index] - mins[index])) * (
                    new_maxs[index] - new_mins[index]) + new_mins[index]
            self.features[index] = v
            index += 1

    def set_label(self, label):
        self.label = label

    def set_filename(self, filename):
        self.filename = filename

    def get_filename(self):
        return self.filename

    def get_features(self):
        return self.features

    def get_feature_at_index(self, index):
        return self.features[index]

    def get_label(self):
        return self.label

    def is_less_than(self, attribute_number, threshold):
        if self.features[attribute_number] <= threshold:
            return True
        return False
