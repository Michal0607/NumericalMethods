import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data=sns.load_dataset('titanic')

features = ['age', 'fare', 'pclass']

data = data.dropna(subset=features)

X = data[features]

X['pclass'] = X['pclass'].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

plt.figure(figsize=(18, 6))

plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=data['survived'], palette='viridis')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Oryginalne dane Titanic')
plt.legend(title='Survived')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data['survived'], palette='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Titanic dataset')

plt.subplot(1, 2, 1)
explained_variance_ratio = pca.explained_variance_ratio_
plt.bar(['PC1', 'PC2'], explained_variance_ratio, color='purple')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance Ratio')

plt.tight_layout()

plt.show()
