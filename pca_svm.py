import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by the first 2 components: {explained_variance}")

svm_model = make_pipeline(PCA(n_components=2), SVC(kernel='linear'))
svm_model.fit(X_pca, y)
y_pred = svm_model.predict(X_pca)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy of model SVM with PCA (2 components): {accuracy:.2f}")

plt.figure(figsize=(16, 10))

plt.subplot(1, 2, 1)
plt.bar(['PC1', 'PC2'], explained_variance, color=['blue', 'green'])
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by First 2 PCA Components')

plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='viridis', edgecolor='k', legend='full')
plt.title("Before PCA")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Class')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', edgecolor='k', legend='full')
plt.title("After PCA")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Class')

plt.tight_layout()
plt.show()
