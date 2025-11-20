import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons

# 1. Make nonlinear 2D data
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)

# 2. Train both models
linear_svm = svm.SVC(kernel="linear", C=1.0)
linear_svm.fit(X, y)

rbf_svm = svm.SVC(kernel="rbf", gamma=0.5, C=1.0)
rbf_svm.fit(X, y)

# 3. Meshgrid for contour drawing
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

# 4. Predict over the grid
Z_linear = linear_svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_rbf = rbf_svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 5. Plot (2 subplots)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_linear, alpha=0.3)
plt.contour(xx, yy, Z_linear, colors="k", linewidths=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors="k")
plt.title("Linear SVM — Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_rbf, alpha=0.3)
plt.contour(xx, yy, Z_rbf, colors="k", linewidths=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors="k")
plt.title("RBF SVM — Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
