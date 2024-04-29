import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sns.set(style="whitegrid")

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

accuracy_before_svd = accuracy_score(y_test, model.predict(X_test))
print(f'Accuracy before SVD: {round(accuracy_before_svd,4)}')

svd = TruncatedSVD(n_components=2)

X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

model.fit(X_train_svd, y_train)

accuracy_after_svd = accuracy_score(y_test, model.predict(X_test_svd))
print(f'Accuracy after SVD: {round(accuracy_after_svd,4)}')