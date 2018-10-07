#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from random_forest import CrossValidationSplitter

if __name__ == "__main__":
    data = pd.read_csv('resources/sonar.all-data.csv', header=None).values
    for n_tree in [1, 4, 16]:
        accuracies = []
        model = None
        splitter = CrossValidationSplitter(data, k_fold=5, rate=0.9)
        for train_data, validate_data in splitter:
            model = RandomForestClassifier(
                n_estimators=n_tree,
                max_depth=5,
                min_samples_leaf=1,
            )
            X,y = train_data[:,:-1], train_data[:,-1]
            model.fit(X,y)
            X_val,y_val = validate_data[:,:-1], validate_data[:,-1]
            accuracies.append(model.score(X,y))
        validation_accuracy = np.mean(accuracies)
        X,y = splitter.test_data[:,:-1], splitter.test_data[:,-1]
        test_accuracy = model.score(X,y)
        print(
            f"Mean cross validation accuracy for {n_tree} trees: {validation_accuracy}")
        print(f"Test accuracy for {n_tree} trees: {test_accuracy}")
