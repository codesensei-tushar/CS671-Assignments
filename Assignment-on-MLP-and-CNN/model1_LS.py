import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DATA LOADING
data_path = {
    "class1.txt": "C:/Users/nshej/dl assignment/CS671_Dataset_Assignment1/Dataset-1/LS/Class1.txt",
    "class2.txt": "C:/Users/nshej/dl assignment/CS671_Dataset_Assignment1/Dataset-1/LS/Class2.txt",
    "class3.txt": "C:/Users/nshej/dl assignment/CS671_Dataset_Assignment1/Dataset-1/LS/Class3.txt",
}

def read_data(file_path):
    data = pd.read_csv(file_path, sep=r"\s+", header=None, engine='python')
    return data

# Concatenate data
data = pd.concat([read_data(data_path["class1.txt"]),
                  read_data(data_path["class2.txt"]),
                  read_data(data_path["class3.txt"])])
data.columns = ['X', 'Y']
data.reset_index(drop=True, inplace=True)
data['label'] = ''
data.loc[0:499, "label"] = "1"
data.loc[500:999, "label"] = "2"
data.loc[1000:1499, "label"] = "3"

# Train-test split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
X_train = train_set[['X', 'Y']].values.astype(float)
Y_train = train_set['label'].values
X_test = test_set[['X', 'Y']].values.astype(float)
Y_test = test_set['label'].values

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP Classifier
class MLP:
    def __init__(self, input_size, hidden_size, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs
        self.weights_input_hidden = np.random.randn(input_size + 1, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size + 1) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        x = np.insert(x, 0, 1)
        self.hidden_input = np.dot(x, self.weights_input_hidden)
        self.hidden_output = self.sigmoid(self.hidden_input)
        hidden_with_bias = np.insert(self.hidden_output, 0, 1)
        self.final_input = np.dot(hidden_with_bias, self.weights_hidden_output)
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                output = self.forward(x)
                error = target - output

                delta_output = error * self.sigmoid_derivative(output)
                hidden_with_bias = np.insert(self.hidden_output, 0, 1)
                delta_hidden = np.dot(self.weights_hidden_output[1:], delta_output) * self.sigmoid_derivative(self.hidden_output)

                self.weights_hidden_output += self.lr * delta_output * hidden_with_bias
                x_with_bias = np.insert(x, 0, 1)
                self.weights_input_hidden += self.lr * np.outer(x_with_bias, delta_hidden)

    def predict(self, x):
        output = self.forward(x)
        return 1 if output >= 0.5 else 0

# OvO training
class_pairs = [("1", "2"), ("1", "3"), ("2", "3")]
models = {}

for class_a, class_b in class_pairs:
    indices = np.where((Y_train == class_a) | (Y_train == class_b))[0]
    X_pair = X_train[indices]
    Y_pair = np.array([1 if label == class_a else 0 for label in Y_train[indices]])
    clf = MLP(input_size=2, hidden_size=4, learning_rate=0.1, epochs=200)
    clf.train(X_pair, Y_pair)
    models[(class_a, class_b)] = clf

# OvO prediction
from collections import Counter
def ovo_predict(x):
    votes = []
    for (class_a, class_b), model in models.items():
        pred = model.predict(x)
        votes.append(class_a if pred == 1 else class_b)
    return Counter(votes).most_common(1)[0][0]

# Test predictions
y_pred = [ovo_predict(x) for x in X_test]

# Accuracy & Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred))
print(f"Accuracy: {accuracy_score(Y_test, y_pred)*100:.2f}%")

# Decision boundary
import matplotlib.colors as mcolors
colors = {'1': 'red', '2': 'blue', '3': 'green'}
cmap = mcolors.ListedColormap(['red', 'blue', 'green'])

def plot_decision_boundary():
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([ovo_predict(x) for x in grid]).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, preds, alpha=0.3, cmap=cmap)
    for label in ['1', '2', '3']:
        subset = X_train[Y_train == label]
        plt.scatter(subset[:, 0], subset[:, 1], label=f"Class {label}", alpha=0.6)
    plt.legend()
    plt.title("Decision Boundary (One-vs-One MLP)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

plot_decision_boundary()
