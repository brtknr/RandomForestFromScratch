#!/usr/bin/env python

import random
import numpy as np
import pandas as pd
from math import sqrt

random.seed(42)
from random_forest import RandomForest, CrossValidationSplitter

def train_test_split(data, rate):
    random.shuffle(data)
    n_train_data = int(len(data) * rate)
    return data[: n_train_data], data[n_train_data:]

if __name__ == "__main__":
    df = pd.read_csv('resources/sonar.all-data.csv', header=None)
    data = df.values.tolist()
    train_data_all, test_data = train_test_split(data, 0.9)

    for n_tree in [1, 5, 25]:
        accuracies = []
        cross_validation_splitter = CrossValidationSplitter(train_data_all, 5)
        model = None
        for train_data, validate_data in cross_validation_splitter:
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
        test_accuracy = model.accuracy(test_data)
        print(f"Mean cross validation accuracy for {n_tree} trees: \
            {validation_accuracy}")
        print(f"Test accuracy for {n_tree} trees: {test_accuracy}")
