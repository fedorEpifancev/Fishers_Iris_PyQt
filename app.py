from PyQt5 import QtWidgets
from PyQt5.QtGui import*
from PyQt5.QtWidgets import*
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import*
import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

i = 0
iris = load_iris()
features = 4
iris_types = ['Setosa', 'Versicolor', 'Verginica']
steps_education = [0]*35
num_steps = [i for i in range(1, 36)]

x_train, x_test, y_train, y_test = train_test_split(iris.data[:, :features], iris.target, test_size = 0.3, shuffle = True)
x_train = torch.FloatTensor(x_train)
x_test =torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class IrisNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden_neurons):
        super(IrisNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, n_hidden_neurons)
        self.activ1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.activ2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        return x
    
    def inference(self, x):
        x = self.forward(x)
        #print(x)
        x = torch.argmax(x, dim=0)
        #print(x)
        return x
    
n_input = 4
n_hidden = 5
iris_net = IrisNet(n_input, n_hidden)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch. optim.Adam(iris_net.parameters(), lr = 0.001)
batch_size = 15

for epoch in range(500):
    order = np.random.permutation(len(x_train))
    for start_index in range(0, len(x_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index+batch_size]

        x_batch = x_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        preds = iris_net.forward(x_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()
        if epoch % 100 == 0:
            test_preds = iris_net.forward(x_test)
            test_preds = test_preds.argmax(dim=1)
            #print((test_preds == y_test).float().mean())
            steps_education[i] = (test_preds == y_test).float().mean().item()
            i = i+1

#print(iris_net.fc1.in_features, np.asarray((test_preds == y_test).float().mean()) > 0.8)
#print(steps_education)

class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.w_width = 500
        self.w_height = 400
        self.setGeometry(200, 200, self.w_width, self.w_height)
        self.setWindowTitle("Определение вида ириса")
        self.UiComponents()
        plt.plot(num_steps, steps_education)
        plt.title('Нейросеть обучилась, прогресс обучения:')
        plt.xlabel('Шаг обучения')
        plt.ylabel('Точность предсказания')
        plt.show()
        self.show()
        
    def UiComponents(self):

        self.label_1 = QLabel("<b>Входные значения:</b>", self)
        self.label_1.move(50, 30)
        self.label_1.adjustSize()
        
        self.label_2 = QLabel("<b>Ответ, вид ириса:</b>", self)
        self.label_2.move(350, 30)
        self.label_2.adjustSize()

        self.btn_result = QPushButton(self)
        self.btn_result.move(50, 250)
        self.btn_result.setText("Ввести значения")
        self.btn_result.clicked.connect(self.btn_clicked)

        self.input1 = QLineEdit(self)
        self.input1.setText("0")
        self.input1.move(210, 60)

        self.label_1 = QLabel("Длина чашелистника:", self)
        self.label_1.move(50, 70)
        self.label_1.adjustSize()

        self.input2 = QLineEdit(self)
        self.input2.setText("0")
        self.input2.move(210, 110)

        self.label_1 = QLabel("Ширина чашелистника:", self)
        self.label_1.move(50, 120)
        self.label_1.adjustSize()

        self.input3 = QLineEdit(self)
        self.input3.setText("0")
        self.input3.move(210, 160)

        self.label_1 = QLabel("Длина лепестка:", self)
        self.label_1.move(50, 170)
        self.label_1.adjustSize()

        self.input4 = QLineEdit(self)
        self.input4.setText("0")
        self.input4.move(210, 210)

        self.label_1 = QLabel("Ширина лепестка:", self)
        self.label_1.move(50, 220)
        self.label_1.adjustSize()

        self.label_3 = QLabel(self)
        self.label_3.move(350, 50)
        self.label_3.setText('нет ответа')
        self.label_3.adjustSize()




    def btn_clicked(self):
        #вызывать внешнюю функцию нейронной сети, определённую перед этим классом
        x1 = torch.Tensor([float(self.input1.text()), 
                           float(self.input2.text()), 
                           float(self.input3.text()), 
                           float(self.input4.text()),
                          ])
        
        self.label_3.setText(iris_types[int((iris_net.inference(x1)).item())])
        #print(steps_education)
        

       
        
        

        #обнуление значений для следующих предсказаний
        del steps_education[:]
        i = 0




        
        

App = QApplication(sys.argv)
window = Window()




    
sys.exit(App.exec_())