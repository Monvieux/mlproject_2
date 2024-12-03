import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PyQt5 import uic
from PyQt5.QtWidgets import *

# Get path of files
pfolder = os.getcwd()
fdataset = os.path.join(pfolder,'src','Datasets','add_2_numbers.csv')
fmodel = os.path.join(pfolder,'src','Models','add_2_numbers')
fui = os.path.join(pfolder,'src','UI','Add two numbers.ui')

print(fui)

data = pd.read_csv(fdataset)
X = data[['x','y']] # features
y = data[['sum']] # target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model,fmodel)

class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(fui, self)
        self.bt_valid.clicked.connect(self.click_button_valid)
        self.show()

    def click_button_valid(self):
        model = joblib.load(fmodel)
        result = model.predict([[float(self.txt_first_number.text()),float(self.txt_second_number.text())]])
        print(float(result.item()))
        self.lbl_result.setText(str(round(result.item(),2)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui()
    window.setFixedSize(window.width(),window.height())
    app.exec_()
