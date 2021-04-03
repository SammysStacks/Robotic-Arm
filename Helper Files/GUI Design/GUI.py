
# --------------------------------------------------------------------------- #

# Basic Modules
import sys
# Import UI Modules
from PyQt5 import QtCore, QtGui, QtWidgets

# --------------------------------------------------------------------------- #


"""gloable variables"""
HomePosition = [-1,-10,5,12,0] # The initial position of robot arm
InitialPosition = [90,90,90,90,90] # Initial position of robot fingers
k = 1 # k=2 for 2k screen, k=1 for 1080 screen
f = 4 # f=0 for 2k screen, f=4 for 1080 screen


'''Gui design'''

class Ui_MainWindow():
    def __init__(self, MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5, initiate=False):
        self.MainWindow = MainWindow
        self.centralwidget = centralwidget
        self.Number_distance = Number_distance
        # Set Up Arms
        self.arm_1 = arm_1
        self.arm_2 = arm_2
        self.arm_3 = arm_3
        self.arm_4 = arm_4
        self.arm_5 = arm_5
        self.translate = QtCore.QCoreApplication.translate

        self.setupUi(self.MainWindow)
        self.MainWindow.show()
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800*k, 750*k)
        
        self.centralwidget.setObjectName("centralwidget")
        
        # Finger Text Font
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        # position input of finger 1 
        numFingers = 5; self.fingerText = [None]*numFingers;
        for fingerInd in range(numFingers):
            # Make Finger
            self.fingerText[fingerInd] = QtWidgets.QTextEdit(self.centralwidget)
            self.fingerText[fingerInd].setGeometry(QtCore.QRect(480*k, (40+80*fingerInd)*k, 121*k, 41*k))
            # Set Finger Text Style
            self.fingerText[fingerInd].setFont(font)
            self.fingerText[fingerInd].setObjectName("textEdit_1")
            self.fingerText[fingerInd].setText(str(InitialPosition[0]))
        
        # button for set finger 1 position
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(630*k, 40*k, 131*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        
        # button for set finger 2 position
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(630*k, 120*k, 131*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15-f)
        font.setBold(True)
        font.setWeight(75)      
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        
        # button for set finger 3 position
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(630*k, 200*k, 131*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        
        # button for set finger 4 position
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(630*k, 280*k, 131*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        
        # button for set finger 5 position
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(630*k, 360*k, 131*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        
        # label for finger 1
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(360*k, 40*k, 91*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        # label for finger 2
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(360*k, 120*k, 91*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        
        # label for finger 3
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(360*k, 200*k, 91*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        
        # label for finger 4
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(360*k, 280*k, 91*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        
        # label for finger 5
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(360*k, 360*k, 91*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        
        # button for hand grab
        self.pushButton_g = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_g.setGeometry(QtCore.QRect(90*k, 50*k, 151*k, 61*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_g.setFont(font)
        self.pushButton_g.setObjectName("pushButton_g")
        
        # button for hand release 
        self.pushButton_r = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_r.setGeometry(QtCore.QRect(90*k, 140*k, 151*k, 61*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_r.setFont(font)
        self.pushButton_r.setObjectName("pushButton_r")
        
        # button for hand reset
        self.pushButton_reset = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_reset.setGeometry(QtCore.QRect(90*k, 230*k, 151*k, 61*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_reset.setFont(font)
        self.pushButton_reset.setObjectName("pushButton_reset")
        
        # position input for arm_1
        self.arm_1.setGeometry(QtCore.QRect(86*k, 478*k, 71*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.arm_1.setFont(font)
        self.arm_1.setObjectName("arm_1")
        self.arm_1.setText(str(HomePosition[0]))
        
        # position input for arm_2
        self.arm_2.setGeometry(QtCore.QRect(186*k, 478*k, 71*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.arm_2.setFont(font)
        self.arm_2.setObjectName("arm_2")
        self.arm_2.setText(str(HomePosition[1]))
        
        # position input for arm_3
        self.arm_3.setGeometry(QtCore.QRect(286*k, 478*k, 71*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.arm_3.setFont(font)
        self.arm_3.setObjectName("arm_3")
        self.arm_3.setText(str(HomePosition[2]))
        
        # position input for arm_4
        self.arm_4.setGeometry(QtCore.QRect(386*k, 478*k, 71*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.arm_4.setFont(font)
        self.arm_4.setObjectName("arm_4")
        self.arm_4.setText(str(HomePosition[3]))
        
        # position input for arm_5
        self.arm_5.setGeometry(QtCore.QRect(486*k, 478*k, 71*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.arm_5.setFont(font)
        self.arm_5.setObjectName("arm_5")
        self.arm_5.setText(str(HomePosition[4]))
        
        # set arm to input location
        self.pushButton_arm = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_arm.setGeometry(QtCore.QRect(600*k, 480*k, 151*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_arm.setFont(font)
        self.pushButton_arm.setObjectName("pushButton_arm")
        
        # label for arm
        self.label_arm = QtWidgets.QLabel(self.centralwidget)
        self.label_arm.setGeometry(QtCore.QRect(20*k, 430*k, 161*k, 31*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setItalic(True)
        self.label_arm.setFont(font)
        self.label_arm.setObjectName("label_arm")
        
        # label for hand
        self.label_hand = QtWidgets.QLabel(self.centralwidget)
        self.label_hand.setGeometry(QtCore.QRect(20*k, 10*k, 191*k, 31*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setItalic(True)
        self.label_hand.setFont(font)
        self.label_hand.setObjectName("label_hand")
        
        # button for move the arm down
        self.pushButton_down = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_down.setGeometry(QtCore.QRect(90*k, 550*k, 171*k, 91*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_down.setFont(font)
        self.pushButton_down.setObjectName("pushButton_down")
        
        # button for move the arm up
        self.pushButton_up = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_up.setGeometry(QtCore.QRect(340*k, 550*k, 171*k, 91*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_up.setFont(font)
        self.pushButton_up.setObjectName("pushButton_up")
        
        # button for move the arm to initial location
        self.Reset_arm = QtWidgets.QPushButton(self.centralwidget)
        self.Reset_arm.setGeometry(QtCore.QRect(580*k, 550*k, 171*k, 91*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-f)
        font.setBold(True)
        font.setWeight(75)
        self.Reset_arm.setFont(font)
        self.Reset_arm.setObjectName("Reset_arm")
        
        # label of distance
        self.label_distance = QtWidgets.QLabel(self.centralwidget)
        self.label_distance.setGeometry(QtCore.QRect(100*k, 380*k, 71*k, 21*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-f)
        self.label_distance.setFont(font)
        self.label_distance.setObjectName("label_distance")
        
        # show the distance
        self.Number_distance.setGeometry(QtCore.QRect(180*k, 380*k, 61*k, 21*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-f)
        self.Number_distance.setFont(font)
        self.Number_distance.setObjectName("Number_distance")
        
        # button for turning laser on
        self.laser_on = QtWidgets.QPushButton(self.centralwidget)
        self.laser_on.setGeometry(QtCore.QRect(60*k, 320*k, 91*k, 31*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-f)
        font.setBold(True)
        font.setWeight(75)
        self.laser_on.setFont(font)
        self.laser_on.setObjectName("laser_on")
        
        # button for turning laser off
        self.laser_off = QtWidgets.QPushButton(self.centralwidget)
        self.laser_off.setGeometry(QtCore.QRect(170*k, 320*k, 91*k, 31*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-f)
        font.setBold(True)
        font.setWeight(75)
        self.laser_off.setFont(font)
        self.laser_off.setObjectName("laser_off")
           
        # set for main window 
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
               

        # link function to button
        self.pushButton_1.clicked.connect(lambda: self.setFingerPos(fingerIndex = 1))
        self.pushButton_2.clicked.connect(lambda: self.setFingerPos(fingerIndex = 2))
        self.pushButton_3.clicked.connect(lambda: self.setFingerPos(fingerIndex = 3))
        self.pushButton_4.clicked.connect(lambda: self.setFingerPos(fingerIndex = 4))
        self.pushButton_5.clicked.connect(lambda: self.setFingerPos(fingerIndex = 5))
        self.pushButton_arm.clicked.connect(self.click_Move_Arm)
        self.laser_on.clicked.connect(self.click_Laser_On)
        self.laser_off.clicked.connect(self.click_Laser_Off)
        
    
    def setFingerPos(self, fingerIndex = 1):
        """
        fingerIndex: Finger Position Starting at the Little Finger (1-Indexed)
        """
        pos = self.fingerText[fingerIndex - 1].toPlainText()
        #self.handController.moveFinger(pos, com_f = 'h' + str(fingerIndex))
            
    def click_Move_Arm(self):
        loc_1 = int(self.arm_1.toPlainText())
        loc_2 = int(self.arm_2.toPlainText())
        loc_3 = int(self.arm_3.toPlainText())
        loc_4 = int(self.arm_4.toPlainText())
        loc_5 = int(self.arm_5.toPlainText())
        #Controller.moveTo([loc_1, loc_2, loc_3, loc_4, loc_5])
        
    def click_Laser_On(self):
        #hand.write(str.encode('s1'))
        return 1
        
    def click_Laser_Off(self):
        #hand.write(str.encode('s0'))
        return 1

  
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(self.translate("MainWindow", "Hand control"))
        self.pushButton_1.setText(self.translate("MainWindow", "Set Position"))
        self.pushButton_2.setText(self.translate("MainWindow", "Set Position"))
        self.pushButton_3.setText(self.translate("MainWindow", "Set Position"))
        self.pushButton_4.setText(self.translate("MainWindow", "Set Position"))
        self.pushButton_5.setText(self.translate("MainWindow", "Set Position"))
        self.label.setText(self.translate("MainWindow", "Finger 1"))
        self.label_2.setText(self.translate("MainWindow", "Finger 2"))
        self.label_3.setText(self.translate("MainWindow", "Finger 3"))
        self.label_4.setText(self.translate("MainWindow", "Finger 4"))
        self.label_5.setText(self.translate("MainWindow", "Finger 5"))
        self.pushButton_g.setText(self.translate("MainWindow", "Grab"))
        self.pushButton_r.setText(self.translate("MainWindow", "Release"))
        self.pushButton_reset.setText(self.translate("MainWindow", "Reset"))
        self.pushButton_arm.setText(self.translate("MainWindow", "Move Arm"))
        self.label_arm.setText(self.translate("MainWindow", "Arm Control"))
        self.label_hand.setText(self.translate("MainWindow", "Hand Control"))
        self.pushButton_down.setText(self.translate("MainWindow", "Move Down"))
        self.pushButton_up.setText(self.translate("MainWindow", "Move Up"))
        self.Reset_arm.setText(self.translate("MainWindow", "Reset Arm"))
        self.label_distance.setText(self.translate("MainWindow", "Distance:"))
        self.Number_distance.setText(self.translate("MainWindow", "no data"))
        self.laser_on.setText(self.translate("MainWindow", "Laser On"))
        self.laser_off.setText(self.translate("MainWindow", "Laser Off"))
    




