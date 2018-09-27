#!/usr/bin/env python

from math import sqrt
import random
import numpy as np
import pandas as pd
import pdb

random.seed(42)


class Node:
    def __init__(self, data):
        # all the data that is held by this node
        self.data = data
        # child nodes
        self.child = dict()
        # category if the current node is a leaf node
        self.category = None
        # get categories
        self.categories = self.__get_categories()
        # calculate number of features
        self.n_features = int(sqrt(len(data[0]) - 1))
        # features
        self.features = self.__get_features()
        # a tuple: (row, column), representing the point where
        # we split the data into the left/right node
        self.split_point = self.__get_split_point() 
        self.left_group, self.right_group = self.__split(*self.split_point)

    def __get_categories(self):
        data = self.data
        return set([row[-1] for row in data])

    def __get_features(self):
        data, n_features = self.data, self.n_features
        n_total_features = len(data[0]) - 1
        features = [i for i in range(n_total_features)]
        random.shuffle(features)
        return features[:n_features]

    def __get_split_point(self):
        data = self.data
        features = self.features
        x, y, gini_index = None, None, None
        for index in range(len(data)):
            for feature in features:
                left, right = self.__split(index, feature)
                current_gini_index = self.__get_gini_index(left, right)
                if gini_index is None or current_gini_index < gini_index:
                    x, y, gini_index = index, feature, current_gini_index
        return x, y

    def __split(self, x, y):
        data = self.data
        split_value = data[x][y]
        left, right = [], []
        for row in data:
            if row[y] <= split_value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def __get_gini_index(self, left, right):
        categories = self.categories
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

    def most_common_category(self):
        data = self.data
        categories = [row[-1] for row in data]
        return max(set(categories), key=categories.count)

class Tree:
    def __init__(self, data, depth, max_depth, min_size):
        self.root = root = Node(data)
        left_group, right_group = root.left_group, root.right_group
        if len(left_group) == 0 or len(right_group) == 0 or depth >= max_depth:
            root.category = root.most_common_category()
        else:
            for i, group in enumerate([left_group, right_group]):
                if len(group) < min_size:
                    root.child[i] = Node(group)
                    root.category = root.most_common_category()
                else:
                    root.child[i] = Tree(group, depth + 1, max_depth, min_size)

    def predict(self, row):
        root = self.root
        if root.category is not None:
            return root.category
        x, y = root.split_point
        split_value = root.data[x][y]
        if row[y] <= split_value:
            return root.child[0].predict(row)
        else:
            return root.child[1].predict(row)


class RandomForest:
    def __init__(self, data, n_trees, max_depth, min_size, n_sample_rate):
        self.trees = trees = []
        for i in range(n_trees):
            random.shuffle(data)
            n_samples = int(len(data) * n_sample_rate)
            tree = Tree(data[: n_samples], 1, max_depth, min_size)
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
        predicted_categories = [self.predict(
            row[:-1]) for row in validate_data]
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
        self.__do_train_test_split()

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_iteration >= self.k_fold:
            raise StopIteration
        self.n_iteration += 1
        return self.__get_train_validate_split()

    def __get_train_validate_split(self):
        data = self.train_validate_data[:]
        random.shuffle(data)
        train_data = data[self.n_train_validate_split:]
        validate_data = data[:self.n_train_validate_split]
        return train_data, validate_data

    def __do_train_test_split(self):
        rate, all_data = self.rate, self.all_data
        random.shuffle(all_data)
        n_train_test_split = int(len(all_data) * rate)
        self.train_validate_data = all_data[: n_train_test_split]
        self.test_data = all_data[n_train_test_split:]
        self.n_train_validate_split = len(
            self.train_validate_data)//self.k_fold


if __name__ == "__main__":
    data = pd.read_csv(
        'resources/sonar.all-data.csv', header=None
    ).values.tolist()
    for n_tree in [1, 4, 16]:
        accuracies = []
        model = None
        splitter = CrossValidationSplitter(data, k_fold=5, rate=0.9)
        #pdb.set_trace()
        for train_data, validate_data in splitter:
            #print(len(data), len(train_data), len(validate_data))
            model = RandomForest(
                data=train_data,
                n_trees=n_tree,
                max_depth=5,
                min_size=1,
                n_sample_rate=0.9
            )
            accuracies.append(model.accuracy(validate_data))
        validation_accuracy = np.mean(accuracies)
        test_accuracy = model.accuracy(splitter.test_data)
        print(
            f"Mean cross validation accuracy for {n_tree} trees: {validation_accuracy}")
        print(f"Test accuracy for {n_tree} trees: {test_accuracy}")
