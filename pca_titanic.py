import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = sns.load_dataset('titanic')

features = ['age', 'fare', 'pclass']

data = data.dropna(subset=features)
X = data[features].copy()

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
plt.title('Original Titanic Data')
plt.legend(title='Survived')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data['survived'], palette='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Titanic Dataset')

plt.subplot(1, 2, 1)
explained_variance_ratio = pca.explained_variance_ratio_
bars = sns.barplot(x=['PC1', 'PC2'], y=explained_variance_ratio, palette='viridis')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance Ratio')

for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Explained variance by PC1:", round(explained_variance_ratio[0], 2))
print("Explained variance by PC2:", round(explained_variance_ratio[1], 2))
print("Total explained variance:", round(explained_variance_ratio.sum(), 2))
