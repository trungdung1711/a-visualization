import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 2 clusters
np.random.seed(42)
n = 200

class0 = np.random.randn(n // 2, 2) + np.array([-2, -2])
class1 = np.random.randn(n // 2, 2) + np.array([2, 2])

X = np.vstack([class0, class1])
y = np.hstack([np.zeros(n // 2), np.ones(n // 2)])

# logistic
log_reg = LogisticRegression()
log_reg.fit(X, y)
w1_lr, w2_lr = log_reg.coef_[0]
b_lr = log_reg.intercept_[0]

# SVM with linear
svm = SVC(kernel="linear")
svm.fit(X, y)
w1_svm, w2_svm = svm.coef_[0]
b_svm = svm.intercept_[0]

plt.figure(figsize=(7, 7))

plt.scatter(class0[:, 0], class0[:, 1])
plt.scatter(class1[:, 0], class1[:, 1])

x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)

# Logistic regression boundary
y_lr = -(w1_lr * x_vals + b_lr) / w2_lr

# SVM boundary
y_svm = -(w1_svm * x_vals + b_svm) / w2_svm

plt.plot(x_vals, y_lr, label="Logistic Regression Boundary")
plt.plot(x_vals, y_svm, label="SVM Boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Logistic Regression vs SVM Decision Boundaries")
plt.legend()

plt.show()
