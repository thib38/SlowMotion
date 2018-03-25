# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MyPictureWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PictureWindow(object):
    def setupUi(self, PictureWindow):
        PictureWindow.setObjectName("PictureWindow")
        PictureWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        PictureWindow.resize(1216, 848)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PictureWindow.sizePolicy().hasHeightForWidth())
        PictureWindow.setSizePolicy(sizePolicy)
        self.CentraWidget = QtWidgets.QWidget(PictureWindow)
        self.CentraWidget.setMinimumSize(QtCore.QSize(1200, 700))
        self.CentraWidget.setObjectName("CentraWidget")
        self.layoutWidget = QtWidgets.QWidget(self.CentraWidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 1202, 825))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.matplotlibWidget = QtWidgets.QWidget(self.layoutWidget)
        self.matplotlibWidget.setMinimumSize(QtCore.QSize(1200, 700))
        self.matplotlibWidget.setMaximumSize(QtCore.QSize(1200, 700))
        self.matplotlibWidget.setObjectName("matplotlibWidget")
        self.verticalLayout.addWidget(self.matplotlibWidget)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(88, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.previousButton = QtWidgets.QPushButton(self.layoutWidget)
        self.previousButton.setObjectName("previousButton")
        self.horizontalLayout.addWidget(self.previousButton)
        spacerItem2 = QtWidgets.QSpacerItem(118, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.playButton = QtWidgets.QPushButton(self.layoutWidget)
        self.playButton.setObjectName("playButton")
        self.horizontalLayout.addWidget(self.playButton)
        spacerItem3 = QtWidgets.QSpacerItem(148, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.pauseButton = QtWidgets.QPushButton(self.layoutWidget)
        self.pauseButton.setObjectName("pauseButton")
        self.horizontalLayout.addWidget(self.pauseButton)
        spacerItem4 = QtWidgets.QSpacerItem(138, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.nextButton = QtWidgets.QPushButton(self.layoutWidget)
        self.nextButton.setObjectName("nextButton")
        self.horizontalLayout.addWidget(self.nextButton)
        spacerItem5 = QtWidgets.QSpacerItem(148, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem6 = QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem6)
        PictureWindow.setCentralWidget(self.CentraWidget)

        self.retranslateUi(PictureWindow)
        QtCore.QMetaObject.connectSlotsByName(PictureWindow)

    def retranslateUi(self, PictureWindow):
        _translate = QtCore.QCoreApplication.translate
        PictureWindow.setWindowTitle(_translate("PictureWindow", "MainWindow"))
        self.previousButton.setText(_translate("PictureWindow", "Previous"))
        self.playButton.setText(_translate("PictureWindow", "Play"))
        self.pauseButton.setText(_translate("PictureWindow", "Pause"))
        self.nextButton.setText(_translate("PictureWindow", "Next"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    PictureWindow = QtWidgets.QMainWindow()
    ui = Ui_PictureWindow()
    ui.setupUi(PictureWindow)
    PictureWindow.show()
    sys.exit(app.exec_())

