import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Ładujemy zbiór danych Titanic z Seaborn
data = sns.load_dataset('titanic')

# Wybieramy cechy do analizy
features = ['age', 'fare', 'pclass']

# Usuwamy wartości brakujące w wybranych cechach
imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Skalujemy dane
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Definiujemy funkcję do redukcji wymiarów przy użyciu SVD
def svd_reduction(X, num_components):
    svd = TruncatedSVD(n_components=num_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd

# Wykonujemy redukcję wymiarów
X_reduced, svd = svd_reduction(X_scaled, 2)

# Wariancja wyjaśniona przez poszczególne komponenty
variance_explained = svd.explained_variance_ratio_
print(f"Wariancja wyjaśniona przez PC 1: {variance_explained[0]:.2f}")
print(f"Wariancja wyjaśniona przez PC 2: {variance_explained[1]:.2f}")
print(f"Łączna wariancja wyjaśniona: {variance_explained.sum():.2f}")

# Ustawienia dla wizualizacji
sns.set(style="whitegrid")

# Tworzymy wykresy
plt.figure(figsize=(12, 6))

# Wykres dla proporcji wariancji wyjaśnionej przez każdy komponent
plt.subplot(1, 2, 1)
sns.barplot(x=['PC 1', 'PC 2'], y=variance_explained, palette='viridis')
plt.ylabel('Proporcja wariancji')
plt.title('Proporcja wariancji wyjaśnionej przez każdy komponent')

# Wykres przed i po SVD
plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=data['survived'], palette='viridis', legend='full')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Przed SVD - Ustandaryzowane dane')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data['survived'], palette='viridis', legend='full')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('SVD - Titanic Dataset')
plt.legend(title='Przeżył', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Dostosowanie układu wykresu
plt.tight_layout()
plt.show()
