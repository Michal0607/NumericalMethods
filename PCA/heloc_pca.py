import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_dataset("mstz/heloc")
data = pd.DataFrame(dataset['train'])

X = data.drop(columns=['is_at_risk'])
y = data['is_at_risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2f}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2f}")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_before_pca = accuracy_score(y_test, predictions)
print(f"Accuracy before PCA: {accuracy_before_pca:.2f}")

model_pca = DecisionTreeClassifier()
model_pca.fit(X_train_pca, y_train)
predictions_pca = model_pca.predict(X_test_pca)
accuracy_after_pca = accuracy_score(y_test, predictions_pca)
print(f"Accuracy after PCA: {accuracy_after_pca:.2f}")

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
bars = sns.barplot(x=['PC 1', 'PC 2'], y=pca.explained_variance_ratio_, palette='viridis')
plt.ylabel('Variance Ratio')
plt.title('Variance Explained by Each Component')

for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height() + 0.01, 
             f'{bar.get_height():.2f}', 
             ha='center', va='bottom', color='black')

plt.subplot(2, 2, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='viridis', legend='full')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Before PCA - Standardized Data')

plt.subplot(2, 2, 4)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', legend='full')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA - Heloc Dataset')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.show()
