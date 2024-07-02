import numpy
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from itertools import product
X = numpy.genfromtxt('..\..\heart_data.csv', delimiter = ',', skip_header = True, deletechars = '?')[:, :-1]
res = numpy.genfromtxt('..\..\heart_data.csv', delimiter = ',', skip_header = True, deletechars = '?')[:, -1]
X1, X2, res1, res2 = sklearn.model_selection.train_test_split(X, res, train_size = 0.7)
si = SimpleImputer()
si.fit(X1)
X1 = si.transform(X1)
X2 = si.transform(X2)
acc = 0
for h, n in product(range(1, 14, 1), range(2, 9, 1)):
    tree = DecisionTreeClassifier(max_depth = h, max_leaf_nodes = n)
    tree.fit(X1, res1)
    pred = tree.predict(X1)
    print(f'глубина дерева = { h }, листья = { n }, точность = { round(sklearn.metrics.accuracy_score(res1, pred), 5) }')
    if sklearn.metrics.accuracy_score(res1, pred) > acc:
        acc = sklearn.metrics.accuracy_score(res1, pred)
        H = h
        N = n
tree = DecisionTreeClassifier(max_depth = H, max_leaf_nodes = N)
tree.fit(X1, res1)
pred = tree.predict(X2)
print(f'Тестовая: глубина дерева = { H }, листья = { N }, точность = { round(sklearn.metrics.accuracy_score(res2, pred), 5) }')