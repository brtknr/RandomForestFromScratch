#!/usr/bin/env python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from random_forest import CrossValidationSplitter

if __name__ == "__main__":
    data = pd.read_csv('resources/sonar.all-data.csv', header=None).values
    for n_trees in [1, 4, 16]:
        cval_accs = []
        test_accs = []
        model = None
        splitter = CrossValidationSplitter(data, k_fold=5, rate=0.9)
        for train_data, validate_data, test_data in splitter:
            model = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=5,
                min_samples_leaf=1,
            )
            X,y = train_data[:,:-1], train_data[:,-1]
            X_val,y_val = validate_data[:,:-1], validate_data[:,-1]
            X_test,y_test = test_data[:,:-1], test_data[:,-1]
            model.fit(X,y)
            cval_accs.append(model.score(X_val,y_val))
            test_accs.append(model.score(X_test,y_test))
        cval_acc = sum(cval_accs)/k_fold
        test_acc = sum(test_accs)/k_fold
        print(
            f"Cross validation accuracy for {n_trees} trees: {cval_acc}")
        print(f"Test accuracy for {n_trees} trees: {test_acc}")
