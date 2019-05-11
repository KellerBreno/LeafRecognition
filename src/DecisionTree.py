class TreeNode:
    def __init__(self, data_set, feature_list, parent=None):
        self.feature_number = None
        self.feature_list = feature_list
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.data_set = data_set
        self.parent = parent

    def train(self):
        if self.data_set.is_uniform():
            label = self.data_set.get_data()[0].get_label()
            leaf = LeafNode(label)
            return leaf

        if len(self.feature_list) == 0:
            labels = self.data_set.get_dataset_statistics()
            best_label = None
            best_frequency = 0
            for key in labels:
                if labels[key] > best_frequency:
                    best_label = key
                    best_frequency = labels[key]
            leaf = LeafNode(best_label)
            return leaf

        current_entropy = self.data_set.get_entropy()
        current_length = self.data_set.get_length()
        information_gain = -1 * float("inf")
        best_feature_index = 0
        best_left_set = None
        best_right_set = None
        best_threshold = 0

        for featureIndex in self.feature_list:
            # threshold = self.data_set.get_best_threshold(featureIndex)
            threshold = self.data_set.get_threshold(featureIndex)

            (leftSet, rightSet) = self.data_set.split_on(featureIndex, threshold)

            left_entropy = leftSet.get_entropy()
            right_entropy = rightSet.get_entropy()
            new_entropy = (leftSet.get_length() / current_length) * left_entropy + (
                    rightSet.get_length() / current_length) * right_entropy
            new_information_gain = current_entropy - new_entropy

            if new_information_gain > information_gain:
                information_gain = new_information_gain
                best_left_set = leftSet
                best_right_set = rightSet
                best_feature_index = featureIndex
                best_threshold = threshold

        new_feature_list = list(self.feature_list)
        new_feature_list.remove(best_feature_index)

        if best_left_set.get_length() == 0 or best_right_set.get_length() == 0:
            labels = self.data_set.get_dataset_statistics()
            best_label = None
            best_frequency = 0

            for key in labels:
                if labels[key] > best_frequency:
                    best_label = key
                    best_frequency = labels[key]
            leaf = LeafNode(best_label)
            return leaf

        self.threshold = best_threshold
        self.feature_number = best_feature_index

        left_child = TreeNode(best_left_set, new_feature_list, self)
        right_child = TreeNode(best_right_set, new_feature_list, self)

        self.left_child = left_child.train()
        self.right_child = right_child.train()

        return self

    def classify(self, register):
        value = register.get_features()[self.feature_number]

        if value < self.threshold:
            return self.left_child.classify(register)
        else:
            return self.right_child.classify(register)


class LeafNode:
    def __init__(self, classification):
        self.label = classification

    def classify(self, register):
        return self.label


class DecisionTree:
    def __init__(self, data):
        self.root_node = None
        self.data = data

    def train(self):
        length = self.data.get_feature_length()
        feature_indices = range(length)
        self.root_node = TreeNode(self.data, feature_indices)
        self.root_node.train()

    def classify(self, register):
        return self.root_node.classify(register)
