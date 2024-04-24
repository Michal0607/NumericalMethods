import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Wczytaj zbiór danych Titanic
data = sns.load_dataset('titanic')

# Wybierz cechy do analizy
features = ['age', 'fare', 'pclass']

# Usuń wiersze z brakującymi wartościami w wybranych cechach
data = data.dropna(subset=features)

# Wyodrębnij wybrane cechy
X = data[features]

# Przekształć cechę 'pclass' na zmienną typu float
X['pclass'] = X['pclass'].astype(float)

# Skaluj dane
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Wykonaj PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Utwórz rysunek z trzema podplotami
plt.figure(figsize=(18, 6))

# Podplot 1: Wykres rozrzutu oryginalnych cech
plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=data['survived'], palette='viridis')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Oryginalne dane Titanic')
plt.legend(title='Survived')

# Podplot 2: Wykres rozrzutu PCA
plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data['survived'], palette='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Titanic dataset')

# Podplot 3: Wykres słupkowy wyjaśnionej wariancji
plt.subplot(1, 2, 1)
explained_variance_ratio = pca.explained_variance_ratio_
plt.bar(['PC1', 'PC2'], explained_variance_ratio, color='purple')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance Ratio')

# Dopasuj rozmieszczenie podplotów
plt.tight_layout()

# Pokaż wykres
plt.show()
