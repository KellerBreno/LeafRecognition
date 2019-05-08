import random
import sys

import numpy as np


class TreeNode:
    def __init__(self, data_set, feature_list, parent=None):
        self.featureNumber = None
        self.featureList = feature_list
        self.threshold = None
        self.leftChild = None
        self.rightChild = None
        self.dataSet = data_set
        self.parent = parent

    def train(self):
        if self.dataSet.is_uniform():
            label = self.dataSet.get_data()[0].get_label()
            leaf = LeafNode(label)
            return leaf

        if len(self.featureList) == 0:
            labels = self.dataSet.get_label_statistics()
            best_label = None
            best_frequency = 0
            for key in labels:
                if labels[key] > best_frequency:
                    best_label = key
                    best_frequency = labels[key]
            leaf = LeafNode(best_label)
            return leaf

        current_entropy = self.dataSet.get_entropy()
        current_length = self.dataSet.get_length()
        information_gain = -1 * float("inf")
        best_feature_index = 0
        best_left_set = None
        best_right_set = None
        best_threshold = 0

        # Feature Bagging, Random subspace
        num = int(np.ceil(np.sqrt(len(self.featureList))))
        feature_subset = random.sample(self.featureList, num)

        for featureIndex in feature_subset:
            threshold = self.dataSet.better_threshold(featureIndex)

            (leftSet, rightSet) = self.dataSet.split_on(featureIndex, threshold)

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

        new_feature_list = list(self.featureList)
        new_feature_list.remove(best_feature_index)

        if best_left_set.get_length() == 0 or best_right_set.get_length() == 0:
            labels = self.dataSet.get_label_statistics()
            best_label = None
            best_frequency = 0

            for key in labels:
                if labels[key] > best_frequency:
                    best_label = key
                    best_frequency = labels[key]
            leaf = LeafNode(best_label)
            return leaf

        self.threshold = best_threshold
        self.featureNumber = best_feature_index

        left_child = TreeNode(best_left_set, new_feature_list, self)
        right_child = TreeNode(best_right_set, new_feature_list, self)

        self.leftChild = left_child.train()
        self.rightChild = right_child.train()

        return self

    def __str__(self):
        return str(self.featureList)

    def __repr__(self):
        return self.__str__()

    def classify(self, sample):
        value = sample.get_features()[self.featureNumber]

        if value < self.threshold:
            return self.leftChild.classify(sample)
        else:
            return self.rightChild.classify(sample)


class LeafNode:
    def __init__(self, classification):
        self.classification = classification

    def classify(self, sample):
        return self.classification


class DecisionTree:
    def __init__(self, data):
        self.rootNode = None
        self.data = data

    def train(self):
        length = self.data.get_feature_length()
        feature_indices = range(length)
        self.rootNode = TreeNode(self.data, feature_indices)
        self.rootNode.train()

    def classify(self, sample):
        return self.rootNode.classify(sample)
