import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sns.set(style="whitegrid")

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000, solver='lbfgs')

model.fit(X_train, y_train)

accuracy_before_pca = accuracy_score(y_test, model.predict(X_test))
print(f'Accuracy before PCA: {round(accuracy_before_pca,4)}')

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model.fit(X_train_pca, y_train)

accuracy_after_pca = accuracy_score(y_test, model.predict(X_test_pca))
print(f'Accuracy after PCA: {round(accuracy_after_pca,4)}')
