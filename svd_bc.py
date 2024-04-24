import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def svd_reduction(X, num_components):
    svd = TruncatedSVD(n_components=num_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_reduced, svd = svd_reduction(X_scaled, 2)

variance_explained = svd.explained_variance_ratio_
print(f"Wariancja wyjaśniona przez PC 1: {variance_explained[0]:.2f}")
print(f"Wariancja wyjaśniona przez PC 2: {variance_explained[1]:.2f}")
print(f"Łączna wariancja wyjaśniona: {variance_explained.sum():.2f}")

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=['PC 1', 'PC 2'], y=svd.explained_variance_ratio_, palette='viridis')
plt.ylabel('Variance Ratio')
plt.title('Variance Explained by Each Component')

plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='viridis', legend='full')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Before SVD - Standardized Data')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette='viridis', legend='full')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('SVD - Breast Cancer Dataset')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.show()

