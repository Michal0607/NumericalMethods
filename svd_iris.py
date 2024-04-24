import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris  # Używamy zestawu danych iris jako zamiennik Glass Dataset, ponieważ Glass Dataset nie jest dostępny bezpośrednio w sklearn

def svd_reduction(X, num_components):
    svd = TruncatedSVD(n_components=num_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd

# Załaduj zestaw danych iris (zamiast Glass Dataset)
data = load_iris()
X = data.data
y = data.target

# Standardyzuj dane
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Zastosuj SVD do zredukowania danych
X_reduced, svd = svd_reduction(X_scaled, 2)

# Wydrukuj wariancję wyjaśnioną przez komponenty
variance_explained = svd.explained_variance_ratio_
print(f"Wariancja wyjaśniona przez PC 1: {variance_explained[0]:.2f}")
print(f"Wariancja wyjaśniona przez PC 2: {variance_explained[1]:.2f}")
print(f"Łączna wariancja wyjaśniona: {variance_explained.sum():.2f}")

# Ustawienia dla Seaborn
sns.set(style="whitegrid")

# Rysowanie wykresów
plt.figure(figsize=(12, 6))

# Wykres słupkowy pokazujący wyjaśnioną wariancję
plt.subplot(1, 2, 1)
sns.barplot(x=['PC 1', 'PC 2'], y=svd.explained_variance_ratio_, palette='viridis')
plt.ylabel('Wariancja')
plt.title('Wariancja wyjaśniona przez każdy komponent')

# Wykres rozproszenia przed SVD - Dane znormalizowane
plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='viridis', legend='full')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Przed SVD - Dane znormalizowane')

# Wykres rozproszenia po SVD - Dane zredukowane
plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette='viridis', legend='full')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('SVD - Iris Dataset')
plt.legend(title='Klasa', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Dostosowanie rozłożenia
plt.tight_layout()
plt.show()
