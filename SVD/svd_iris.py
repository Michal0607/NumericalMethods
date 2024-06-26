import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris 

def svd_reduction(X, num_components):
    svd = TruncatedSVD(n_components=num_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd

data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_reduced, svd = svd_reduction(X_scaled, 2)

variance_explained = svd.explained_variance_ratio_
print(f"Explained variance by PC 1: {variance_explained[0]:.2f}")
print(f"Explained variance by PC 2: {variance_explained[1]:.2f}")
print(f"Total explained variance: {variance_explained.sum():.2f}")

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
bars=sns.barplot(x=['PC 1', 'PC 2'], y=svd.explained_variance_ratio_, palette='flare')
plt.ylabel('Variance')
plt.title('Explained Variance Ratio')

for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=[target_names[i] for i in y], palette='flare', legend='full')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Before SVD - Standardized Data')
plt.legend(title='Species')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=[target_names[i] for i in y], palette='flare', legend='full')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('After SVD - Iris Dataset')
plt.legend(title='Species', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
