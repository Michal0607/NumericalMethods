import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def pca(X, num_components):
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    return X_reduced

data = load_iris()
X = data.data
y = data.target

X_reduced = pca(X, 2)

sns.set(style="whitegrid")

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)  
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="viridis")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Iris Data (First two features)')
plt.legend(title='Species')

plt.subplot(1, 2, 2)  
scatter = sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="viridis", legend=None)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Iris dataset')

plt.tight_layout()
plt.show()
