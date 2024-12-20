# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f:\ML_PROJECTS\mlproject_2\src\UI\Health_insurance_cost.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(495, 271)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lbl_title = QtWidgets.QLabel(self.centralwidget)
        self.lbl_title.setGeometry(QtCore.QRect(0, 0, 491, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_title.setFont(font)
        self.lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_title.setObjectName("lbl_title")
        self.bt_valid = QtWidgets.QPushButton(self.centralwidget)
        self.bt_valid.setGeometry(QtCore.QRect(350, 40, 101, 23))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.bt_valid.setFont(font)
        self.bt_valid.setObjectName("bt_valid")
        self.lbl_result = QtWidgets.QLabel(self.centralwidget)
        self.lbl_result.setGeometry(QtCore.QRect(316, 80, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_result.setFont(font)
        self.lbl_result.setText("")
        self.lbl_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_result.setObjectName("lbl_result")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(10, 40, 141, 181))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.lbl_age = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lbl_age.setFont(font)
        self.lbl_age.setObjectName("lbl_age")
        self.lbl_gender = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lbl_gender.setFont(font)
        self.lbl_gender.setObjectName("lbl_gender")
        self.lbl_bmi = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lbl_bmi.setFont(font)
        self.lbl_bmi.setObjectName("lbl_bmi")
        self.lbl_children = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lbl_children.setFont(font)
        self.lbl_children.setObjectName("lbl_children")
        self.lbl_smoker = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lbl_smoker.setFont(font)
        self.lbl_smoker.setObjectName("lbl_smoker")
        self.lbl_region = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lbl_region.setFont(font)
        self.lbl_region.setObjectName("lbl_region")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(170, 40, 135, 181))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.txt_age = QtWidgets.QLineEdit(self.widget)
        self.txt_age.setObjectName("txt_age")
        self.verticalLayout.addWidget(self.txt_age)
        self.cbx_gender = QtWidgets.QComboBox(self.widget)
        self.cbx_gender.setObjectName("cbx_gender")
        self.cbx_gender.addItem("")
        self.cbx_gender.addItem("")
        self.cbx_gender.addItem("")
        self.verticalLayout.addWidget(self.cbx_gender)
        self.txt_bmi = QtWidgets.QLineEdit(self.widget)
        self.txt_bmi.setObjectName("txt_bmi")
        self.verticalLayout.addWidget(self.txt_bmi)
        self.txt_children = QtWidgets.QLineEdit(self.widget)
        self.txt_children.setObjectName("txt_children")
        self.verticalLayout.addWidget(self.txt_children)
        self.cbx_smoker = QtWidgets.QComboBox(self.widget)
        self.cbx_smoker.setObjectName("cbx_smoker")
        self.cbx_smoker.addItem("")
        self.cbx_smoker.addItem("")
        self.cbx_smoker.addItem("")
        self.verticalLayout.addWidget(self.cbx_smoker)
        self.cbx_region = QtWidgets.QComboBox(self.widget)
        self.cbx_region.setObjectName("cbx_region")
        self.cbx_region.addItem("")
        self.cbx_region.addItem("")
        self.cbx_region.addItem("")
        self.cbx_region.addItem("")
        self.cbx_region.addItem("")
        self.verticalLayout.addWidget(self.cbx_region)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 495, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.cbx_gender.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Health Insurance Cost"))
        self.lbl_title.setText(_translate("MainWindow", "Health Insurance Cost Prediction"))
        self.bt_valid.setText(_translate("MainWindow", "Predict"))
        self.lbl_age.setText(_translate("MainWindow", "Enter Your Age :"))
        self.lbl_gender.setText(_translate("MainWindow", "Gender :"))
        self.lbl_bmi.setText(_translate("MainWindow", "Enter Your BMI :"))
        self.lbl_children.setText(_translate("MainWindow", "Number of Children :"))
        self.lbl_smoker.setText(_translate("MainWindow", "Smoker :"))
        self.lbl_region.setText(_translate("MainWindow", "Enter Your Region :"))
        self.cbx_gender.setItemText(0, _translate("MainWindow", "Select Item"))
        self.cbx_gender.setItemText(1, _translate("MainWindow", "Female"))
        self.cbx_gender.setItemText(2, _translate("MainWindow", "Male"))
        self.cbx_smoker.setItemText(0, _translate("MainWindow", "Select Item"))
        self.cbx_smoker.setItemText(1, _translate("MainWindow", "No"))
        self.cbx_smoker.setItemText(2, _translate("MainWindow", "Yes"))
        self.cbx_region.setItemText(0, _translate("MainWindow", "Select Item"))
        self.cbx_region.setItemText(1, _translate("MainWindow", "southwest"))
        self.cbx_region.setItemText(2, _translate("MainWindow", "southeast"))
        self.cbx_region.setItemText(3, _translate("MainWindow", "northwest"))
        self.cbx_region.setItemText(4, _translate("MainWindow", "northeast"))
