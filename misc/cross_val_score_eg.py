from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:300]
y = diabetes.target[:300]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=10))  # doctest: +ELLIPSIS

