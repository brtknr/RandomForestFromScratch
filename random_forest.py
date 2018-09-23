#!/usr/bin/env python

import random
import numpy as np
import pandas as pd
from math import sqrt

random.seed(42)

class Node:
    def __init__(self, data):

        # all the data that is held by this node
        self.data = data

        # left child node
        self.left = None

        # right child node
        self.right = None

        # category if the current node is a leaf node
        self.category = None

        # a tuple: (row, column), representing the point where we split the
        # data into the left/right node
        self.split_point = None

    def set_most_common_category(self):
        data = self.data
        categories = [row[-1] for row in data]
        self.category = max(set(categories), key=categories.count)

class Tree:
    def __init__(self, data, depth, max_depth, min_size, n_features):
        self.data, self.depth, self.max_depth, self.min_size, self.n_features \
            = data, depth, max_depth, min_size, n_features
        self.root = root = Node(data)
        x, y = self.get_split_point()
        left_group, right_group = self.split(x, y)
        if len(left_group) == 0 or len(right_group) == 0 or depth >= max_depth:
            root.set_most_common_category()
        else:
            root.split_point = (x, y)
            if len(left_group) < min_size:
                root.left = Node(left_group)
                root.left.set_most_common_category()
            else:
                root.left = Tree(left_group, depth + 1, max_depth, min_size, n_features)

            if len(right_group) < min_size:
                root.right = Node(right_group)
                root.right.set_most_common_category()
            else:
                root.right = Tree(right_group, depth + 1, max_depth, min_size, n_features)

    def get_features(self):
        data, n_features = self.data, self.n_features
        n_total_features = len(data[0]) - 1
        features = [i for i in range(n_total_features)]
        random.shuffle(features)
        return features[:n_features]

    def get_split_point(self):
        data = self.data
        features = self.get_features()
        x, y, gini_index = None, None, None
        for index in range(len(data)):
            for feature in features:
                left, right = self.split(index, feature)
                current_gini_index = self.get_gini_index(left, right)
                if gini_index is None or current_gini_index < gini_index:
                    x, y, gini_index = index, feature, current_gini_index
        return x, y

    def split(self, x, y):
        data = self.data
        split_value = data[x][y]
        left, right = [], []
        for row in data:
            if row[y] <= split_value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_categories(self):
        data = self.data
        return set([row[-1] for row in data])

    def get_gini_index(self, left, right):
        categories = self.get_categories()
        gini_index = 0
        for group in left, right:
            if len(group) == 0:
                continue
            score = 0
            for category in categories:
                p = [row[-1] for row in group].count(category) / len(group)
                score += p * p
            gini_index += (1 - score) * (len(group) / len(left + right))
        return gini_index

    def predict(self, row):
        root = self.root
        if root.category is not None:
            return root.category
        x, y = root.split_point
        split_value = root.data[x][y]
        if row[y] <= split_value:
            return root.left.predict(row)
        else:
            return root.right.predict(row)


class RandomForest:
    def __init__(self, data, n_trees, max_depth, min_size, n_features, n_sample_rate):
        self.data, self.n_trees, self.max_depth, self.min_size \
            = data, n_trees, max_depth, min_size
        self.n_features, self.n_sample_rate = n_features, n_sample_rate
        self.trees = trees = []
        for i in range(n_trees):
            random.shuffle(data)
            n_samples = int(len(data) * n_sample_rate)
            tree = Tree(data[: n_samples], 1, max_depth, min_size, n_features)
            trees.append(tree)

    def predict(self, row):
        trees = self.trees
        prediction = []
        for tree in trees:
            prediction.append(tree.predict(row))
        return max(set(prediction), key=prediction.count)

    def accuracy(self, validate_data):
        n_total = 0
        n_correct = 0
        predicted_categories = [self.predict(row[:-1]) for row in validate_data]
        correct_categories = [row[-1] for row in validate_data]
        for predicted_category, correct_category in zip(predicted_categories, correct_categories):
            n_total += 1
            if predicted_category == correct_category:
                n_correct += 1
        return n_correct / n_total


class CrossValidationSplitter:
    def __init__(self, all_data, k_fold, rate):
        self.all_data, self.k_fold, self.rate = all_data, k_fold, rate
        self.n_iteration = 0
        self.train_data, self.test_data = self.train_test_split()
        self.n_batch = (1 / self.k_fold) * len(self.train_data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_iteration >= self.k_fold:
            raise StopIteration
        self.n_iteration += 1
        return self.load_data()

    def load_data(self):
        data_copy = self.train_data[:]
        train_data = []
        while len(train_data) < self.n_batch:
            train_data.append(self.pop_random_row(data_copy))
        validate_data = data_copy
        return train_data, validate_data

    def pop_random_row(self, data):
        random.shuffle(data)
        return data[0]

    def train_test_split(self):
        rate, all_data = self.rate, self.all_data
        random.shuffle(all_data)
        n_train_data = int(len(all_data) * rate)
        return all_data[: n_train_data], all_data[n_train_data:]


if __name__ == "__main__":
    df = pd.read_csv('resources/sonar.all-data.csv', header=None)
    data = df.values.tolist()
    for n_tree in [1, 5, 25]:
        accuracies = []
        model = None
        splitter = CrossValidationSplitter(data, k_fold=5, rate=0.9)
        for train_data, validate_data in splitter:
            n_features = int(sqrt(len(train_data[0]) - 1))
            model = RandomForest(
                data=train_data,
                n_trees=n_tree,
                max_depth=5,
                min_size=1,
                n_features=n_features,
                n_sample_rate=0.9
            )
            accuracies.append(model.accuracy(validate_data))
        validation_accuracy = np.mean(accuracies)
        test_accuracy = model.accuracy(splitter.test_data)
        print(f"Mean cross validation accuracy for {n_tree} trees: {validation_accuracy}")
        print(f"Test accuracy for {n_tree} trees: {test_accuracy}")
