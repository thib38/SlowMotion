# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'removeNoP.ui'
#
# Created by: PyQt5 UI code generator 5.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_RemoveNoPQDialog(object):
    def setupUi(self, RemoveNoPQDialog):
        RemoveNoPQDialog.setObjectName("RemoveNoPQDialog")
        RemoveNoPQDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        RemoveNoPQDialog.resize(396, 129)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(RemoveNoPQDialog.sizePolicy().hasHeightForWidth())
        RemoveNoPQDialog.setSizePolicy(sizePolicy)
        RemoveNoPQDialog.setModal(True)
        self.buttonBox = QtWidgets.QDialogButtonBox(RemoveNoPQDialog)
        self.buttonBox.setGeometry(QtCore.QRect(90, 90, 261, 32))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(RemoveNoPQDialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 341, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.textBrowser = QtWidgets.QTextBrowser(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout.addWidget(self.textBrowser)
        self.spinBox_N = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spinBox_N.setFont(font)
        self.spinBox_N.setMinimum(1)
        self.spinBox_N.setMaximum(9)
        self.spinBox_N.setObjectName("spinBox_N")
        self.horizontalLayout.addWidget(self.spinBox_N)
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_2.sizePolicy().hasHeightForWidth())
        self.textBrowser_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textBrowser_2.setFont(font)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.horizontalLayout.addWidget(self.textBrowser_2)
        self.spinBox_P = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spinBox_P.setFont(font)
        self.spinBox_P.setMinimum(2)
        self.spinBox_P.setMaximum(10)
        self.spinBox_P.setObjectName("spinBox_P")
        self.horizontalLayout.addWidget(self.spinBox_P)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(RemoveNoPQDialog)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(20, 50, 321, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textBrowser_3.setFont(font)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.horizontalLayout_2.addWidget(self.textBrowser_3)
        self.spinBox_NbRow = QtWidgets.QSpinBox(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spinBox_NbRow.setFont(font)
        self.spinBox_NbRow.setMinimum(2)
        self.spinBox_NbRow.setMaximum(5000)
        self.spinBox_NbRow.setObjectName("spinBox_NbRow")
        self.horizontalLayout_2.addWidget(self.spinBox_NbRow)
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_4.sizePolicy().hasHeightForWidth())
        self.textBrowser_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textBrowser_4.setFont(font)
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.horizontalLayout_2.addWidget(self.textBrowser_4)

        self.retranslateUi(RemoveNoPQDialog)
        self.buttonBox.accepted.connect(RemoveNoPQDialog.accept)
        self.buttonBox.rejected.connect(RemoveNoPQDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(RemoveNoPQDialog)

    def retranslateUi(self, RemoveNoPQDialog):
        _translate = QtCore.QCoreApplication.translate
        RemoveNoPQDialog.setWindowTitle(_translate("RemoveNoPQDialog", "Dialog"))
        self.textBrowser.setHtml(_translate("RemoveNoPQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Remove</p></body></html>"))
        self.textBrowser_2.setHtml(_translate("RemoveNoPQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> rows out of </p></body></html>"))
        self.textBrowser_3.setHtml(_translate("RemoveNoPQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">for the next </p></body></html>"))
        self.textBrowser_4.setHtml(_translate("RemoveNoPQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> rows</p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    RemoveNoPQDialog = QtWidgets.QDialog()
    ui = Ui_RemoveNoPQDialog()
    ui.setupUi(RemoveNoPQDialog)
    RemoveNoPQDialog.show()
    sys.exit(app.exec_())

