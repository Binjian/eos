# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/99.sandbox.ipynb.

# %% auto 0
__all__ = ['irises', 'iris_inputs', 'iris_prediction', 'import_data', 'fit_dt_model', 'make_irs_prediction',
           'convert_to_int_list']

# %% ../nbs/99.sandbox.ipynb 3
from sklearn import datasets, tree
import numpy as np

# %% ../nbs/99.sandbox.ipynb 4
def import_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

# %% ../nbs/99.sandbox.ipynb 5
def fit_dt_model() -> tree.DecisionTreeClassifier:
    X, y = import_data()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    return clf

# %% ../nbs/99.sandbox.ipynb 6
def make_irs_prediction(X_values: np.array) -> tuple:
    model = fit_dt_model()
    class_pred = model.predict(X_values)
    pred_proba = model.predict_proba(X_values)
    return class_pred, pred_proba

# %% ../nbs/99.sandbox.ipynb 7
def convert_to_int_list(number_strings: list) -> list:
    list_of_int_lists = []
    int_list = []
    for number_string in number_strings:
        int_list = []
        for number in number_string.split(','):
            int_list.append(float(number))
        list_of_int_lists.append(int_list)
    return list_of_int_lists
   

# %% ../nbs/99.sandbox.ipynb 8
irises = ["5.4, 3.7, 1.5, 0.2"]
iris_inputs = convert_to_int_list(irises)

iris_prediction = []
for iris in iris_inputs:
    class_pred, class_probas = make_irs_prediction(np.array([iris]))
    iris_prediction.append([class_pred[0], class_probas[0][class_pred[0]]])
print(iris_prediction)