import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def pca(X, num_components):
    # Centrowanie danych
    X_meaned = X - np.mean(X, axis=0)
    
    # Obliczanie macierzy kowariancji
    cov_mat = np.cov(X_meaned, rowvar=False)
    
    # Obliczanie wartości i wektorów własnych
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    
    # Sortowanie wektorów własnych zgodnie z wartościami własnymi
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    
    # Wybór podzbioru z posortowanych wektorów własnych
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    
    # Transformacja danych
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    
    return X_reduced

data = load_iris()
X = data.data
y = data.target

# Redukcja do 2 komponentów dla celów wizualizacji
X_reduced = pca(X, 2)

# Wizualizacja
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Iris dataset')
plt.colorbar(scatter)
plt.show()