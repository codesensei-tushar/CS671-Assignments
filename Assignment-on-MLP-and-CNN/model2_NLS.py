import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# DATA LOADING
dataset = pd.read_csv("nls.csv")
dataset['label'] = dataset['label'].map({1: 0, 2: 1, 3: 2})

X = dataset[['X', 'Y']].values.astype(float)
y = dataset['label'].values

# ONE-HOT ENCODING
def one_hot_encode(y):
    n_classes = len(np.unique(y))
    one_hot = np.zeros((len(y), n_classes))
    for i, val in enumerate(y):
        one_hot[i, val] = 1
    return one_hot

Y = one_hot_encode(y)

# DATA SPLITTING
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# NORMALIZATION
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SINGLE MLP CLASSIFIER
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            # Forward
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.relu(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.softmax(Z2)

            # Backward
            dZ2 = A2 - Y
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.relu_derivative(Z1)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0)

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

    def predict(self, x):
        Z1 = np.dot(x, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return np.argmax(A2, axis=1)

# ONE-VS-ONE CLASSIFIER
class OvOMLPClassifier:
    def __init__(self, input_size, hidden_size, learning_rate, epochs):
        self.classifiers = {}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, Y):
        n_classes = Y.shape[1]
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                idx = np.where((Y[:, i] == 1) | (Y[:, j] == 1))[0]
                X_ij = X[idx]
                y_ij = Y[idx]
                y_binary = np.array([1 if y[i] == 1 else 0 for y in y_ij])
                y_binary_onehot = one_hot_encode(y_binary)
                mlp = MLP(self.input_size, self.hidden_size, 2, self.learning_rate)
                mlp.train(X_ij, y_binary_onehot, self.epochs)
                self.classifiers[(i, j)] = mlp

    def predict(self, x):
        votes = [0] * 3
        for (i, j), clf in self.classifiers.items():
            pred = clf.predict(np.array([x]))[0]
            votes[i if pred == 1 else j] += 1
        return np.argmax(votes) + 1

# PLOT DECISION BOUNDARY
def plot_decision_boundary(model, X, Y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model.predict(point) for point in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    colors = ['r', 'g', 'b']
    labels = ['1', '2', '3']
    for idx, label in enumerate(labels):
        class_points = X[np.array([np.argmax(y) == idx for y in Y])]
        plt.scatter(class_points[:, 0], class_points[:, 1], c=colors[idx], label=f'Class {label}', s=10)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# TRAINING OvO CLASSIFIER
ovo = OvOMLPClassifier(input_size=2, hidden_size=4, learning_rate=0.1, epochs=100)
ovo.train(X_train, Y_train)

# TESTING
def test_ovo_model(model, X_test, Y_test):
    correct = 0
    total = len(X_test)
    for i in range(total):
        predicted_class = model.predict(X_test[i])
        actual_class = np.argmax(Y_test[i]) + 1
        print(f"Sample {i + 1}: Feature: {X_test[i]}, Predicted: {predicted_class}, Actual: {actual_class}")
        if predicted_class == actual_class:
            correct += 1
    accuracy = (correct / total) * 100
    print(f"\nOvO Model Accuracy: {accuracy:.2f}%")

test_ovo_model(novo, X_test, Y_test)

# CONFUSION MATRIX
y_pred_ovo = [novo.predict(x) for x in X_test]
y_true_ovo = [np.argmax(y) + 1 for y in Y_test]
conf_matrix = confusion_matrix(y_true_ovo, y_pred_ovo)
accuracy = accuracy_score(y_true_ovo, y_pred_ovo)

print("OvO Confusion Matrix:")
print(conf_matrix)
print(f"OvO Classification Accuracy: {accuracy * 100:.2f}%")

# PLOT DECISION BOUNDARY
plot_decision_boundary(novo, X_train, Y_train, title="OvO MLP Decision Boundary")
