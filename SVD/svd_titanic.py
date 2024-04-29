import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def svd_reduction(X, num_components):
    svd = TruncatedSVD(n_components=num_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd

data = sns.load_dataset('titanic')

features = ['age', 'fare', 'pclass']

imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

X_reduced, svd = svd_reduction(X_scaled, 2)

variance_explained = svd.explained_variance_ratio_
print(f"Explained variance by PC 1: {variance_explained[0]:.2f}")
print(f"Explained variance by PC 2: {variance_explained[1]:.2f}")
print(f"Total explained variance: {variance_explained.sum():.2f}")

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
bars=sns.barplot(x=['PC 1', 'PC 2'], y=variance_explained, palette='flare')
plt.ylabel('Variance Explained')
plt.title('Explained Variance Ratio')

for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=data['survived'], palette='flare', legend='full')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Original Titanic Data - Before SVD')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data['survived'], palette='flare', legend='full')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Titanic Data - After SVD')

plt.tight_layout()
plt.show()
