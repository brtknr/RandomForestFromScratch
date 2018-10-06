#!/usr/bin/env python

from math import sqrt
import random
import pandas as pd
import pdb
import asyncio

random.seed(42)


class Tree:
    def __init__(self, depth, max_depth, min_size):
        self.depth, self.max_depth, self.min_size = depth, max_depth, min_size
        # category if the current node is a leaf node
        self.leaf_category = None
        # child nodes
        self.child = dict()

    async def fit(self, data):
        depth, max_depth, min_size = self.depth, self.max_depth, self.min_size
        # get categories
        categories = [row[-1] for row in data]
        self.unique_categories = set(categories)
        # features
        n_features = len(data[0]) - 1
        features = self.__get_subset_features(n_features)
        # a tuple: (row, column), representing the point where
        # we split the data into the left/right node
        self.split_point = self.__get_split_point(data, features)
        left, right = self.__split(data, *self.split_point)
        jobs = []
        for i, group in enumerate([left, right]):
            if len(group) < min_size or depth >= max_depth:
                # Most common category
                self.leaf_category = max(self.unique_categories, key=categories.count)
            else:
                child = self.child[i] = Tree(depth + 1, max_depth, min_size)
                jobs.append(child.fit(group))
        await asyncio.gather(*jobs)

    def __get_subset_features(self, n_features):
        n_subset = int(sqrt(n_features))
        features = list(range(n_features))
        random.shuffle(features)
        return features[:n_subset]

    def __get_split_point(self, data, features):
        val, x, y, gi = None, None, None, None
        for index in range(len(data)):
            for feature in features:
                this_val = data[index][feature]
                left, right = self.__split(data, this_val, index, feature)
                this_gi = self.__get_gini_index(left, right)
                if gi is None or this_gi < gi:
                    val, x, y, gi = this_val, index, feature, this_gi
        return val, x, y

    def __split(self, data, val, x, y):
        left, right = [], []
        for row in data:
            if row[y] <= val:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def __get_gini_index(self, left, right):
        gini_index = 0
        for group in left, right:
            if len(group) == 0:
                continue
            score = 0
            for category in self.unique_categories:
                p = [row[-1] for row in group].count(category) / len(group)
                score += p * p
            gini_index += (1 - score) * (len(group) / len(left + right))
        return gini_index

    def predict(self, row):
        if self.leaf_category:
            return self.leaf_category
        val, x, y = self.split_point
        index = 0 if row[y] <= val else 1
        return self.child[index].predict(row)


class RandomForest:
    def __init__(self, n_trees, n_sample_rate, max_depth, min_size):
        self.n_sample_rate = n_sample_rate
        self.trees = trees = []
        for i in range(n_trees):
            tree = Tree(1, max_depth, min_size)
            trees.append(tree)

    async def fit(self, data):
        n_samples = int(len(data) * self.n_sample_rate)
        jobs = []
        for tree in self.trees:
            random.shuffle(data)
            jobs.append(tree.fit(data[: n_samples]))
        await asyncio.gather(*jobs)

    def predict(self, row):
        trees = self.trees
        prediction = []
        for tree in trees:
            prediction.append(tree.predict(row))
        return max(set(prediction), key=prediction.count)

    def accuracy(self, data):
        n_correct = 0
        pairs = [(self.predict(row[:-1]), row[-1]) for row in data]
        n_correct = sum([predicted == actual for predicted, actual in pairs])
        return n_correct / len(data)


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
    for n_trees in [1, 4, 16]:
        accuracies = []
        model = None
        splitter = CrossValidationSplitter(data, k_fold=5, rate=0.9)
        #pdb.set_trace()
        for train_data, validate_data in splitter:
            #print(len(data), len(train_data), len(validate_data))
            model = RandomForest(
                n_trees=n_trees,
                n_sample_rate=0.9,
                max_depth=5,
                min_size=1,
            )
            asyncio.run(model.fit(data=train_data))
            accuracies.append(model.accuracy(data=validate_data))
        cv_accuracy = sum(accuracies)/n_trees
        test_accuracy = model.accuracy(splitter.test_data)
        print(f"Cross validation accuracy for {n_trees} trees: {cv_accuracy}")
        print(f"Test accuracy for {n_trees} trees: {test_accuracy}")
