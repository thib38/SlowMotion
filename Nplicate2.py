# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Nplicate2.ui'
#
# Created by: PyQt5 UI code generator 5.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_NplicateQDialog(object):
    def setupUi(self, NplicateQDialog):
        NplicateQDialog.setObjectName("NplicateQDialog")
        NplicateQDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        NplicateQDialog.resize(448, 298)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(NplicateQDialog.sizePolicy().hasHeightForWidth())
        NplicateQDialog.setSizePolicy(sizePolicy)
        NplicateQDialog.setModal(True)
        self.buttonBox = QtWidgets.QDialogButtonBox(NplicateQDialog)
        self.buttonBox.setGeometry(QtCore.QRect(110, 250, 261, 32))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(NplicateQDialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 341, 73))
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
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(NplicateQDialog)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(-20, 90, 444, 75))
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_NbRow.sizePolicy().hasHeightForWidth())
        self.spinBox_NbRow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spinBox_NbRow.setFont(font)
        self.spinBox_NbRow.setMinimum(1)
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
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(NplicateQDialog)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(30, 190, 321, 73))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textBrowser_5.setFont(font)
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.horizontalLayout_3.addWidget(self.textBrowser_5)
        self.duplicateMethodComboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget_3)
        self.duplicateMethodComboBox.setObjectName("duplicateMethodComboBox")
        self.horizontalLayout_3.addWidget(self.duplicateMethodComboBox)

        self.retranslateUi(NplicateQDialog)
        self.buttonBox.accepted.connect(NplicateQDialog.accept)
        self.buttonBox.rejected.connect(NplicateQDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(NplicateQDialog)

    def retranslateUi(self, NplicateQDialog):
        _translate = QtCore.QCoreApplication.translate
        NplicateQDialog.setWindowTitle(_translate("NplicateQDialog", "Dialog"))
        self.textBrowser.setHtml(_translate("NplicateQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Duplicate</p></body></html>"))
        self.textBrowser_2.setHtml(_translate("NplicateQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> times</p></body></html>"))
        self.textBrowser_3.setHtml(_translate("NplicateQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">next </p></body></html>"))
        self.textBrowser_4.setHtml(_translate("NplicateQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> rows</p></body></html>"))
        self.textBrowser_5.setHtml(_translate("NplicateQDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">duplicate method</p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    NplicateQDialog = QtWidgets.QDialog()
    ui = Ui_NplicateQDialog()
    ui.setupUi(NplicateQDialog)
    NplicateQDialog.show()
    sys.exit(app.exec_())

