import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score

#DATA LOADING

data_path={
    "class1.txt":"C:/Users/nshej/dl assignment/CS671_Dataset_Assignment1/Dataset-1/LS/Class1.txt",
    "class2.txt":"C:/Users/nshej/dl assignment/CS671_Dataset_Assignment1/Dataset-1/LS/Class2.txt",
    "class3.txt":"C:/Users/nshej/dl assignment/CS671_Dataset_Assignment1/Dataset-1/LS/Class3.txt",
}
def read_data(file_path):
    data = pd.read_csv(file_path, sep=r"\s+", header=None,engine='python')
    return data
read_data(data_path["class1.txt"])
read_data(data_path["class2.txt"])
read_data(data_path["class3.txt"])


#concatenating all the data
data = pd.concat([read_data(data_path["class1.txt"]), read_data(data_path["class2.txt"]), read_data(data_path["class3.txt"])])
data.columns = ['X', 'Y']
data.reset_index(drop=True, inplace=True)
data['label'] = ''  # Create the 'label' column
data.loc[0:500,"label"]="1"
data.loc[500:1000,"label"]="2"
data.loc[1000:1500,"label"]="3"
data["class1"]=0
data["class2"]=0
data["class3"]=0
data.loc[data["label"]=="1","class1"]=1
data.loc[data["label"]=="2","class2"]=1
data.loc[data["label"]=="3","class3"]=1
data=data.drop('label',axis=1)
data["label"]=data[["class1","class2","class3"]].values.tolist()
dataset1=pd.DataFrame(data)


#TRIAN_TEST_SPLIT


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
train_set,test_set=train_test_split(dataset1,test_size=0.2,random_state=42)
# print(train_set)
# print(test_set)
X_train=np.array(train_set[['X','Y']].values)
X_train=X_train.astype(float)
Y_train=np.array(train_set["label"].tolist())

#TO AVOID OVERFITTING

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train=scaler.transform(X_train)

#MODEL

class MLP:
    def __init__(self,input_size,hidden_size,num_classes,learning_rate=0.01,epochs=1000):
        self.input_size=input_size 
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.lr=learning_rate
        self.epochs=epochs

        self.weights_input_hidden=np.random.randn(input_size+1,hidden_size)*0.01
        self.weights_hidden_output=np.random.randn(hidden_size+1,num_classes)*0.01
        
        self.loss_history=[]
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sigmoid_derivative(self,x):
        return x*(1-x)
    
    def softmax(self, x):
        """Softmax Function for Output Layer"""
        exp_x = np.exp(x - np.max(x))  # Numerical stability this give good result
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    def forward(self,x):
        x=np.insert(x,0,1)
        self.hidden_input=np.dot(x,self.weights_input_hidden)
        self.hidden_output=self.sigmoid(self.hidden_input)

        hidden_with_bias=np.insert(self.hidden_output,0,1)

        self.final_input=np.dot(hidden_with_bias,self.weights_hidden_output)
        self.final_output = self.softmax(self.final_input)

        
        return self.final_output
    
    def train(self,X,y):
        for epoch in range(self.epochs):
            total_loss=0
            for i in range(len(X)):
                x=X[i]
                target=np.array(y[i])

                output=self.forward(x)
                error=target-output
                loss=np.sum(error**2)
                total_loss+=loss

                #Backpropagation

                hidden_with_bias = np.insert(self.hidden_output, 0, 1)
                delta_output = error * self.sigmoid_derivative(output)
                delta_hidden = np.dot(self.weights_hidden_output[1:], delta_output) * self.sigmoid_derivative(self.hidden_output)

                self.weights_hidden_output += self.lr * np.outer(hidden_with_bias, delta_output)
                x_with_bias = np.insert(x, 0, 1)
                self.weights_input_hidden += self.lr * np.outer(x_with_bias, delta_hidden) 
            # print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss}")
                

            avg_loss = total_loss / len(X)
            # print(avg_loss)
            self.loss_history.append(avg_loss)
    
    def predict(self, x):
        """Prediction Function"""
        output = self.forward(x)
        return np.argmax(output) + 1 
    def plot_loss(self):
        plt.plot(range(1,len(self.loss_history)+1),self.loss_history,marker=0,linestyle="-")
        plt.xlabel("epochs")
        plt.ylabel("Average Loss")
        plt.grid(True)
        plt.show()

mlp = MLP(input_size=2, hidden_size=4, num_classes=3, learning_rate=0.1, epochs=100)
mlp.train(X_train, Y_train)
mlp.plot_loss()


from sklearn.preprocessing import StandardScaler


X_test=test_set[['X','Y']].values
X_test=X_test.astype(float)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_test=scaler.transform(X_test)
Y_test=test_set["label"].values



def test_model(perceptron, X_test, Y_test):
    correct = 0
    total = len(X_test)

    for i in range(total):
        predicted_class= mlp.predict(X_test[i])  # Get class prediction
        actual_class = np.argmax(Y_test[i]) + 1  # Convert one-hot to class label

        print(f"Sample {i+1}: Feature: {X_test[i]}, Predicted: {predicted_class}, Actual: {actual_class}")

        if predicted_class == actual_class:
            correct += 1

    accuracy = (correct / total) * 100
    print(f"\nModel Accuracy: {accuracy:.2f}%")
test_model(MLP, X_test, Y_test)


#INFERENCES

from sklearn.metrics import confusion_matrix,accuracy_score
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = np.array([model.predict(np.array([xx_val, yy_val])) for xx_val, yy_val in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.show()

# Plot Decision Boundaries for each class pair
plot_decision_boundary(mlp, X_train, Y_train, "Decision Boundary - Training Data")

# Confusion Matrix & Accuracy
y_pred = [mlp.predict(x) for x in X_test]
y_true = [np.argmax(y) + 1 for y in Y_test]

conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")