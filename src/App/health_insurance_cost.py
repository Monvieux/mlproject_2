import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor)
from sklearn.metrics import r2_score

from PyQt5 import uic
from PyQt5.QtWidgets import *

def evaluate_models(X_train, y_train, X_test, y_test, models:dict):
    report={}
    for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        report[list(models.keys())[i]] = test_model_score

    return report

# Get path of files
pfolder = os.getcwd()
fdataset = os.path.join(pfolder,'src','Datasets','health_insurance_cost.csv')
fmodel = os.path.join(pfolder,'src','Models','health_insurance_cost')
fui = os.path.join(pfolder,'src','UI','health_insurance_cost.ui')

# test if model exist
if os.path.exists(fmodel)!=True:
    data = pd.read_csv(fdataset)

    data['sex'] = data['sex'].map({'female':0, 'male':1})
    data['smoker'] = data['smoker'].map({'yes':1, 'no':0})
    data['region'] = data['region'].map({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4})
    X = data.drop(['charges'], axis=1)
    y = data['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression':LinearRegression(),
        'SVM':SVR(),
        'RandomForestRegressor':RandomForestRegressor(),
        'GradientBoostingRegressor':GradientBoostingRegressor(),
    }

    # Test all models in list models
    model_report:dict=evaluate_models(X_train, y_train, X_test, y_test,models)

    # To get best model score from dict
    best_model_score = max(sorted(model_report.values()))

    # To gest best model name from dict
    best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
    best_model = models[best_model_name]
    best_model.fit(X,y)
    joblib.dump(best_model,fmodel)


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(fui, self)
        self.bt_valid.clicked.connect(self.click_button_valid)
        self.show()

    def click_button_valid(self):
        model = joblib.load(fmodel)
        f1 = int(self.txt_age.text())
        f2 = int(self.cbx_gender.currentIndex())-1
        f3 = float(self.txt_bmi.text())
        f4 = float(self.txt_children.text())
        f5 = int(self.cbx_smoker.currentIndex())-1
        f6 = int(self.cbx_region.currentIndex())-1

        result = model.predict([[f1,f2,f3,f4,f5,f6]])
        self.lbl_result.setText(str(round(result.item(),2)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui()
    window.setFixedSize(window.width(),window.height())
    app.exec_()
