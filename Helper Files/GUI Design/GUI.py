
# --------------------------------------------------------------------------- #

# Basic Modules
import sys
import time
# Import UI Modules
from PyQt5 import QtCore, QtGui, QtWidgets

# Imort Robotic Control Files
sys.path.append('./Helper Files/Robotic Control/')  # Folder with All the Helper Files
import moveRobot as robotController              # Functions to Control Robot Movement

# --------------------------------------------------------------------------- #

class Ui_MainWindow():
    def __init__(self, handArduino = None, numFingers = 5, numActuators = 5):
        # Start GUI
        self.app = QtWidgets.QApplication(sys.argv)
        
        # Basic Initial Variables
        self.HomePosition = [0, 5, 8, -5, -3]     # The initial position of robot arm
        self.InitialPosition = [90,90,90,90,90]  # Initial position of robot fingers
        self.k = 1 # k=2 for 2k screen, k=1 for 1080 screen
        self.f = 4 # f=0 for 2k screen, f=4 for 1080 screen
        
        # Robot Parameters
        self.robotControl = None
        self.numFingers = numFingers
        self.numActuators = numActuators
        self.handArduino = handArduino
    
        # Create UI
        self.MainWindow = QtWidgets.QMainWindow()
        self.translate = QtCore.QCoreApplication.translate
        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.Number_distance = QtWidgets.QLabel(self.centralwidget)
        
        # Setup and Display UI
        self.setupUi()
        self.MainWindow.show()
    
    def initiateRoboticMovement(self):
        # Initiate Robot Class
        self.robotControl = robotController.robotControl(handArduino = self.handArduino, guiApp = self)
        
        # Link the Gesture Buttons to Their Respective Functions
        self.pushButton_g.clicked.connect(lambda: self.robotControl.moveLeft())
        self.pushButton_r.clicked.connect(lambda: self.robotControl.moveRight())
        self.pushButton_down.clicked.connect(lambda: self.robotControl.moveDown())
        self.pushButton_up.clicked.connect(lambda: self.robotControl.moveUp())
        self.Reset_arm.clicked.connect(lambda: self.robotControl.goHome())

    def setupUi(self):
        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(800*self.k, 750*self.k)
        self.centralwidget.setObjectName("centralwidget")
        
        # Finger Text Font
        fontFingerText = QtGui.QFont()
        fontFingerText.setFamily("Arial")
        fontFingerText.setPointSize(15)
        self.fingerText = [None]*self.numFingers;
        
        # Push Button Font
        fontFingerButton = QtGui.QFont()
        fontFingerButton.setFamily("Arial")
        fontFingerButton.setPointSize(15-self.f)
        fontFingerButton.setBold(True)
        fontFingerButton.setWeight(75)
        self.fingerButton = [None]*self.numFingers;
        
        # Finger Label Font
        fontFingerLabel = QtGui.QFont()
        fontFingerLabel.setFamily("Arial")
        fontFingerLabel.setPointSize(16-self.f)
        fontFingerLabel.setBold(True)
        fontFingerLabel.setWeight(75)
        self.fingerLabel = [None]*self.numFingers;
                
        for fingerInd in range(self.numFingers):
            # Make Finger Text Object
            self.fingerText[fingerInd] = QtWidgets.QTextEdit(self.centralwidget)
            self.fingerText[fingerInd].setGeometry(QtCore.QRect(480*self.k, (40+80*fingerInd)*self.k, 121*self.k, 41*self.k))
            # Set Finger Text Style
            self.fingerText[fingerInd].setFont(fontFingerText)
            self.fingerText[fingerInd].setObjectName("textEdit_" + str(fingerInd+1))
            self.fingerText[fingerInd].setText(str(self.InitialPosition[0]))
        
            # Make Finger Button Object
            self.fingerButton[fingerInd] = QtWidgets.QPushButton(self.centralwidget)
            self.fingerButton[fingerInd].setGeometry(QtCore.QRect(630*self.k, (40+80*fingerInd)*self.k, 131*self.k, 41*self.k))
            # Set Finger Button Style
            self.fingerButton[fingerInd].setFont(fontFingerButton)
            self.fingerButton[fingerInd].setObjectName("pushButton_" + str(fingerInd+1))
        
            # Make Finger Label
            self.fingerLabel[fingerInd] = QtWidgets.QLabel(self.centralwidget)
            self.fingerLabel[fingerInd].setGeometry(QtCore.QRect(360*self.k, (40+80*fingerInd)*self.k, 91*self.k, 41*self.k))
            # Set Finger Label Style
            self.fingerLabel[fingerInd].setFont(fontFingerLabel)
            self.fingerLabel[fingerInd].setObjectName("label_" + str(fingerInd+1))
        
        # Arm Position Font
        fontArm = QtGui.QFont()
        fontArm.setFamily("Arial")
        fontArm.setPointSize(16-self.f)
        fontArm.setBold(True)
        fontArm.setWeight(75)
        self.armPos = [None]*self.numActuators;
        
        for armInd in range(self.numActuators):
            # Make Robot Actuator Arm Text Object
            self.armPos[armInd] = QtWidgets.QTextEdit(self.centralwidget)
            self.armPos[armInd].setGeometry(QtCore.QRect((86+100*armInd)*self.k, 478*self.k, 71*self.k, 41*self.k))
            # Make Robot Actuator Arm Text Style
            self.armPos[armInd].setFont(fontArm)
            self.armPos[armInd].setObjectName("arm_" + str(armInd+1))
            self.armPos[armInd].setText(str(self.HomePosition[armInd]))
        
        
        # Hand Grab/Release Button
        fontGrab = QtGui.QFont()
        fontGrab.setFamily("Arial")
        fontGrab.setPointSize(22-self.f)
        fontGrab.setBold(True)
        fontGrab.setWeight(75)
        # button for hand grab
        self.pushButton_g = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_g.setGeometry(QtCore.QRect(90*self.k, 50*self.k, 151*self.k, 61*self.k))
        self.pushButton_g.setFont(fontGrab)
        self.pushButton_g.setObjectName("pushButton_g")
        
        # button for hand release 
        self.pushButton_r = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_r.setGeometry(QtCore.QRect(90*self.k, 140*self.k, 151*self.k, 61*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_r.setFont(font)
        self.pushButton_r.setObjectName("pushButton_r")
        
        # button for hand reset
        self.pushButton_reset = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_reset.setGeometry(QtCore.QRect(90*self.k, 230*self.k, 151*self.k, 61*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_reset.setFont(font)
        self.pushButton_reset.setObjectName("pushButton_reset")
        
        
        # set arm to input location
        self.pushButton_arm = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_arm.setGeometry(QtCore.QRect(600*self.k, 480*self.k, 151*self.k, 41*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_arm.setFont(font)
        self.pushButton_arm.setObjectName("pushButton_arm")
        
        # label for arm
        self.label_arm = QtWidgets.QLabel(self.centralwidget)
        self.label_arm.setGeometry(QtCore.QRect(20*self.k, 430*self.k, 161*self.k, 31*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-self.f)
        font.setItalic(True)
        self.label_arm.setFont(font)
        self.label_arm.setObjectName("label_arm")
        
        # label for hand
        self.label_hand = QtWidgets.QLabel(self.centralwidget)
        self.label_hand.setGeometry(QtCore.QRect(20*self.k, 10*self.k, 191*self.k, 31*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-self.f)
        font.setItalic(True)
        self.label_hand.setFont(font)
        self.label_hand.setObjectName("label_hand")
        
        # button for move the arm down
        self.pushButton_down = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_down.setGeometry(QtCore.QRect(90*self.k, 550*self.k, 171*self.k, 91*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_down.setFont(font)
        self.pushButton_down.setObjectName("pushButton_down")
        
        # button for move the arm up
        self.pushButton_up = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_up.setGeometry(QtCore.QRect(340*self.k, 550*self.k, 171*self.k, 91*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_up.setFont(font)
        self.pushButton_up.setObjectName("pushButton_up")
        
        # button for move the arm to initial location
        self.Reset_arm = QtWidgets.QPushButton(self.centralwidget)
        self.Reset_arm.setGeometry(QtCore.QRect(580*self.k, 550*self.k, 171*self.k, 91*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.Reset_arm.setFont(font)
        self.Reset_arm.setObjectName("Reset_arm")
        
        # label of distance
        self.label_distance = QtWidgets.QLabel(self.centralwidget)
        self.label_distance.setGeometry(QtCore.QRect(100*self.k, 380*self.k, 71*self.k, 21*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-self.f)
        self.label_distance.setFont(font)
        self.label_distance.setObjectName("label_distance")
        
        # show the distance
        self.Number_distance.setGeometry(QtCore.QRect(180*self.k, 380*self.k, 61*self.k, 21*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-self.f)
        self.Number_distance.setFont(font)
        self.Number_distance.setObjectName("Number_distance")
        
        # button for turning laser on
        self.laser_on = QtWidgets.QPushButton(self.centralwidget)
        self.laser_on.setGeometry(QtCore.QRect(60*self.k, 320*self.k, 91*self.k, 31*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.laser_on.setFont(font)
        self.laser_on.setObjectName("laser_on")
        
        # button for turning laser off
        self.laser_off = QtWidgets.QPushButton(self.centralwidget)
        self.laser_off.setGeometry(QtCore.QRect(170*self.k, 320*self.k, 91*self.k, 31*self.k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12-self.f)
        font.setBold(True)
        font.setWeight(75)
        self.laser_off.setFont(font)
        self.laser_off.setObjectName("laser_off")
           
        # set for main window 
        self.MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self.MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)
               

        # Link the Finger Buttons
        self.fingerButton[0].clicked.connect(lambda: self.setFingerPos(fingerIndex = 0 + 1))
        self.fingerButton[1].clicked.connect(lambda: self.setFingerPos(fingerIndex = 1 + 1))
        self.fingerButton[2].clicked.connect(lambda: self.setFingerPos(fingerIndex = 2 + 1))
        self.fingerButton[3].clicked.connect(lambda: self.setFingerPos(fingerIndex = 3 + 1))
        self.fingerButton[4].clicked.connect(lambda: self.setFingerPos(fingerIndex = 4 + 1))
        
        # Link the Operational Buttons
        self.pushButton_arm.clicked.connect(self.click_Move_Arm)
        self.laser_on.clicked.connect(self.click_Laser_On)
        self.laser_off.clicked.connect(self.click_Laser_Off)
        self.pushButton_reset.clicked.connect(self.resetButton)
    
    def resetButton(self):
        print("Resetting")
        for fingerInd in range(self.numFingers):
            time.sleep(2)
            self.setFingerPos(fingerIndex = fingerInd+1, pos = "90")
    
    def setFingerPos(self, fingerIndex = 1, pos = None):
        """
        fingerIndex: Finger Position Starting at the Little Finger (1-Indexed)
        """
        if not pos:
            pos = self.fingerText[fingerIndex - 1].toPlainText()
        # Move the Finger if the Position is Within the Interval [0, 180]
        if float(pos) < 0 or float(pos) > 180:
            print("Will Not Move the Finger Outside of [0, 180] Interval")
        else:
            self.robotControl.moveFinger(pos, com_f = 'h' + str(fingerIndex))
        
    def click_Move_Arm(self):
        pos = list(map(lambda posList: int(posList.toPlainText()), self.armPos))
        self.robotControl.moveTo(pos)
        
    def click_Laser_On(self):
        if self.handArduino:
            self.handArduino.write(str.encode('s1'))
        
    def click_Laser_Off(self):
        if self.handArduino:
            self.handArduino.write(str.encode('s0'))
        self.Number_distance.setText(self.translate("MainWindow", "Laser Off"))

  
    def retranslateUi(self):
        self.MainWindow.setWindowTitle(self.translate("MainWindow", "Hand control"))
        for fingerInd in range(self.numFingers):
            self.fingerButton[fingerInd].setText(self.translate("MainWindow", "Set Position"))
            self.fingerLabel[fingerInd].setText(self.translate("MainWindow", "Finger " + str(fingerInd+1)))
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
    




