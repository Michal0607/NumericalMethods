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
    return X_reduced, sorted_eigenvalue

data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

X_reduced, eigen_values = pca(X, 2)

total_variance = np.sum(eigen_values)
explained_variance_ratio = eigen_values / total_variance
explained_variance_by_pc = explained_variance_ratio[:2]

sns.set(style="whitegrid")

plt.figure(figsize=(12, 12))

plt.subplot(1, 2, 1)
sns.barplot(x=['PC1', 'PC2'], y=explained_variance_by_pc, palette="viridis")
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Each PC')

plt.subplot(2, 2, 2)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=[target_names[i] for i in y], palette="viridis")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Iris Data (First two features)')
plt.legend(title='Species')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=[target_names[i] for i in y], palette="viridis")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Iris Dataset')
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.show()

print("Explained variance by PC1:", round(explained_variance_by_pc[0],2))
print("Explained variance by PC2:", round(explained_variance_by_pc[1],2))
print("Total explained variance:", round(explained_variance_by_pc.sum(),2))
