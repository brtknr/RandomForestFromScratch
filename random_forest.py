#!/usr/bin/env python

import numpy as np
import pandas as pd
import pdb

np.random.seed(42)


class Tree:
    def __init__(self, max_depth, min_size, depth=1):
        self.depth, self.max_depth, self.min_size = depth, max_depth, min_size

    def fit(self, data):
        depth, max_depth, min_size = self.depth, self.max_depth, self.min_size
        # all the data that is held by this node
        self.data = data
        # child nodes
        self.child = dict()
        # category if the current node is a leaf node
        self.category = None
        # get categories
        self.categories = self.__get_categories()
        # features
        self.features = self.__get_features()
        # a tuple: (row, column), representing the point where
        # we split the data into the left/right node
        self.split_point = self.__get_split_point() 
        self.left_group, self.right_group = self.__split(*self.split_point)
        left_group, right_group = self.left_group, self.right_group
        if len(left_group) == 0 or len(right_group) == 0 or depth >= max_depth:
            self.category = self.most_common_category()
        else:
            for i, this_group in enumerate([left_group, right_group]):
                self.child[i] = child = Tree(max_depth, min_size, depth+1)
                if len(this_group) < min_size:
                    self.category = self.most_common_category()
                else:
                    child.fit(this_group)

    def __get_categories(self):
        return set(self.data[:,-1])

    def __get_features(self):
        data = self.data
        n_all_features = data.shape[1] - 1
        n_features = int(np.sqrt(n_all_features))
        return np.random.choice(n_all_features, n_features, replace=False)

    def __get_split_point(self):
        data = self.data
        features = self.features
        x, y, gini_index = None, None, None
        for index in range(len(data)):
            #pdb.set_trace()
            for feature in features:
                left, right = self.__split(index, feature)
                current_gini_index = self.__get_gini_index(left, right)
                if gini_index is None or current_gini_index < gini_index:
                    x, y, gini_index = index, feature, current_gini_index
        return x, y

    def __split(self, x, y):
        data = self.data
        condition = data[:,y] > data[x, y]
        left, right = [], []
        for cond, row in zip(condition, data):
            if cond:
                right.append(row)
            else:
                left.append(row)
        return np.array(left), np.array(right)

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
                #pdb.set_trace()
            gini_index += (1 - score) * (len(group) / (len(left) + len(right)))
        return gini_index

    def most_common_category(self):
        data = self.data
        categories = [row[-1] for row in data]
        return max(set(categories), key=categories.count)

    def predict(self, row):
        if self.category is not None:
            return self.category
        x, y = self.split_point
        split_value = self.data[x][y]
        if row[y] <= split_value:
            return self.child[0].predict(row)
        else:
            return self.child[1].predict(row)


class RandomForest:
    def __init__(self, n_trees, max_depth, min_size, n_sample_rate):
        self.n_sample_rate = n_sample_rate
        self.trees = [Tree(max_depth, min_size) for i in range(n_trees)]

    def fit(self, data):
        n_sample = int(len(data) * self.n_sample_rate)
        for tree in self.trees:
            tree.fit(data[np.random.choice(len(data), n_sample)])

    def predict(self, row):
        trees = self.trees
        prediction = [tree.predict(row) for tree in trees]
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
        np.random.shuffle(data)
        train_data = data[self.n_train_validate_split:]
        validate_data = data[:self.n_train_validate_split]
        return train_data, validate_data

    def __do_train_test_split(self):
        rate, all_data = self.rate, self.all_data
        np.random.shuffle(all_data)
        perm = np.random.permutation(len(all_data))
        n_train_test_split = int(len(all_data) * rate)
        self.train_validate_data = all_data[: n_train_test_split]
        self.test_data = all_data[n_train_test_split:]
        self.n_train_validate_split = len(
            self.train_validate_data)//self.k_fold


if __name__ == "__main__":
    data = pd.read_csv('resources/sonar.all-data.csv', header=None).values
    for n_tree in [1, 4, 16]:
        accuracies = []
        model = None
        splitter = CrossValidationSplitter(data, k_fold=5, rate=0.9)
        for train_data, validate_data in splitter:
            #pdb.set_trace()
            print(len(data), len(train_data), len(validate_data))
            model = RandomForest(
                n_trees=n_tree,
                max_depth=5,
                min_size=1,
                n_sample_rate=0.9
            )
            model.fit(train_data)
            accuracies.append(model.accuracy(validate_data))
        validation_accuracy = np.mean(accuracies)
        test_accuracy = model.accuracy(splitter.test_data)
        print(
            f"Mean cross validation accuracy for {n_tree} trees: {validation_accuracy}")
        print(f"Test accuracy for {n_tree} trees: {test_accuracy}")
