from src.DecisionTree import DecisionTree


class RandomForest(object):
    def __init__(self, data, number_trees=100):
        self.data = data
        self.number_trees = number_trees
        self.forest = []
        for i in range(number_trees):
            subset = data.get_random_subset()
            self.forest.append(DecisionTree(subset))

    def train(self):
        for tree in self.forest:
            tree.train()

    def classify(self, register):
        votes = {}
        for tree in self.forest:
            label = tree.classify(register)
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1
        best_label = None
        best_frequency = 0
        for key in votes:
            if votes[key] > best_frequency:
                best_label = key
                best_frequency = votes[key]
        return best_label

    def classify_all(self, data):
        answers = []
        for register in data.get_data():
            answers.append((register.filename, self.classify(register)))
        return answers
