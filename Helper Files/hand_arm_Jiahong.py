#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import time
import functools
import threading

import serial
import serial.tools.list_ports

sys.path.append('home/pi/')

import innfos

from PyQt5 import QtCore, QtGui, QtWidgets


# In[ ]:


"""gloable variables"""
HomePosition = [-1,-10,5,12,0] # The initial position of robot arm
InitialPosition = [90,90,90,90,90] # Initial position of robot fingers
k = 1 # k=2 for 2k screen, k=1 for 1080 screen
f = 4 # f=0 for 2k screen, f=4 for 1080 screen
distance_slow = 100 # robot arm will slow down if the distance is less than this number,
speed_slow = 0.02 # speed of arm in slow mode
speed_fast = 0.15# speed of arm in fast mode
speed_x = 1 # speed_x = 1 when the arm is in fast mode, otherwise, speed_x = 0


# In[ ]:


'''Gui design'''
_translate = QtCore.QCoreApplication.translate

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
        # Set Up GUI Window
        if initiateRobot:
            self.setupUi(self.MainWindow)
            self.MainWindow.show()
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800*k, 750*k)
        
        self.centralwidget.setObjectName("centralwidget")
        
        # position input of finger 1 
        self.textEdit_1 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_1.setGeometry(QtCore.QRect(480*k, 40*k, 121*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        self.textEdit_1.setFont(font)
        self.textEdit_1.setObjectName("textEdit_1")
        self.textEdit_1.setText(str(InitialPosition[0]))
        
        # position input of finger 2 
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(480*k, 120*k, 121*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)        
        self.textEdit_2.setFont(font)
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_2.setText(str(InitialPosition[1]))
        
        # position input of finger 3
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(480*k, 200*k, 121*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        self.textEdit_3.setFont(font)
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_3.setText(str(InitialPosition[2]))
        
        # position input of finger 4 
        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(480*k, 280*k, 121*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        self.textEdit_4.setFont(font)
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_4.setText(str(InitialPosition[3]))
        
        # position input of finger 5
        self.textEdit_5 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_5.setGeometry(QtCore.QRect(480*k, 360*k, 121*k, 41*k))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        self.textEdit_5.setFont(font)
        self.textEdit_5.setObjectName("textEdit_5")
        self.textEdit_5.setText(str(InitialPosition[4]))
        
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
        self.pushButton_1.clicked.connect(self.click_Set_Position_1)
        self.pushButton_2.clicked.connect(self.click_Set_Position_2)
        self.pushButton_3.clicked.connect(self.click_Set_Position_3)
        self.pushButton_4.clicked.connect(self.click_Set_Position_4)
        self.pushButton_5.clicked.connect(self.click_Set_Position_5)
        self.pushButton_arm.clicked.connect(self.click_Move_Arm)
        self.laser_on.clicked.connect(self.click_Laser_On)
        self.laser_off.clicked.connect(self.click_Laser_Off)
        
        
    def click_Set_Position_1(self):
        pos = self.textEdit_1.toPlainText()
        finger_1(pos)
        
    def click_Set_Position_2(self):
        pos = self.textEdit_2.toPlainText()
        finger_2(pos)  
                
    def click_Set_Position_3(self):
        pos = self.textEdit_3.toPlainText()
        finger_3(pos)    
        
    def click_Set_Position_4(self):
        pos = self.textEdit_4.toPlainText()
        finger_4(pos) 
        
    def click_Set_Position_5(self):
        pos = self.textEdit_5.toPlainText()
        finger_5(pos)
            
    def click_Move_Arm(self):
        loc_1 = int(self.arm_1.toPlainText())
        loc_2 = int(self.arm_2.toPlainText())
        loc_3 = int(self.arm_3.toPlainText())
        loc_4 = int(self.arm_4.toPlainText())
        loc_5 = int(self.arm_5.toPlainText())
        Controller.moveTo([loc_1, loc_2, loc_3, loc_4, loc_5])
        
    def click_Laser_On(self):
        hand.write(str.encode('s1'))
        
    def click_Laser_Off(self):
        hand.write(str.encode('s0'))

        
        
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Hand control"))
        self.pushButton_1.setText(_translate("MainWindow", "Set Position"))
        self.pushButton_2.setText(_translate("MainWindow", "Set Position"))
        self.pushButton_3.setText(_translate("MainWindow", "Set Position"))
        self.pushButton_4.setText(_translate("MainWindow", "Set Position"))
        self.pushButton_5.setText(_translate("MainWindow", "Set Position"))
        self.label.setText(_translate("MainWindow", "Finger 1"))
        self.label_2.setText(_translate("MainWindow", "Finger 2"))
        self.label_3.setText(_translate("MainWindow", "Finger 3"))
        self.label_4.setText(_translate("MainWindow", "Finger 4"))
        self.label_5.setText(_translate("MainWindow", "Finger 5"))
        self.pushButton_g.setText(_translate("MainWindow", "Grab"))
        self.pushButton_r.setText(_translate("MainWindow", "Release"))
        self.pushButton_reset.setText(_translate("MainWindow", "Reset"))
        self.pushButton_arm.setText(_translate("MainWindow", "Move Arm"))
        self.label_arm.setText(_translate("MainWindow", "Arm Control"))
        self.label_hand.setText(_translate("MainWindow", "Hand Control"))
        self.pushButton_down.setText(_translate("MainWindow", "Move Down"))
        self.pushButton_up.setText(_translate("MainWindow", "Move Up"))
        self.Reset_arm.setText(_translate("MainWindow", "Reset Arm"))
        self.label_distance.setText(_translate("MainWindow", "Distance:"))
        self.Number_distance.setText(_translate("MainWindow", "no data"))
        self.laser_on.setText(_translate("MainWindow", "Laser On"))
        self.laser_off.setText(_translate("MainWindow", "Laser Off"))
    
def distanceRead():
    if hand.in_waiting > 20:
        _ = hand.read_all
    if hand.in_waiting > 0:
        d_laser = hand.read_until()
        distance = d_laser.decode()
        ui.Number_distance.setText(_translate("MainWindow", str(distance)))
        distance = int(distance)
        global speed_x
        print(distance, distance_slow)
        if distance < distance_slow and speed_x == 1:
            Controller.updateMovementParams([speed_slow, speed_slow, speed_slow, speed_slow, speed_slow], 'speed')
            speed_x = 0
            print('slow')
        elif distance >= distance_slow and speed_x == 0:
            Controller.updateMovementParams([speed_fast, speed_fast, speed_fast, speed_fast, speed_fast], 'speed')
            speed_x = 1
            print('fast')
                       
    threading.Timer(0.05,distanceRead).start()


# In[ ]:


class initiateRobot(Ui_MainWindow):
    def __init__(self, MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5):
        # Load Parent Class
        super().__init__(MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5)
        
        # Label Actuators
        self.actuID = [0x01, 0x02, 0x03, 0x04, 0x05]
        
        # Define Common Positions
        self.HomePos = [-1, -10, 5, 12, 0] # Set the Start/End Home Position
        self.FancyPos = [-1, -14, -8, 13, 0] # Set the Start/End Home Position
        self.RestPos = [-1.2004829049110413, 0.2907487154006958, 1.065185546875, 1.10516357421875, 0.00054931640625] # Set the Start/End Home Position
        self.posError = 0.001
        
        # Define Movement Parameters
        # Define Movement Parameters
        self.maxSpeed = [0.1, 0.1, 0.1, 0.1, 0.1]
        self.accel = [1.5, 1.5, 1.5, 1.5, 1.5]
        self.decel = [-1.5, -1.5, -1.5, -1.5, -1.5]
        #actuID = innfos.queryID(6)
    
    def setRoboParams(self):
        # Find and Connect to Actuators
        innfos.enableact(self.actuID)
        innfos.trapposmode(self.actuID)
        time.sleep(0.5)
        
        # Set Speed, Acceleration, and Deceleration
        innfos.trapposset(self.actuID, self.accel, self.maxSpeed, self.decel)
        time.sleep(0.5)
        
        # Set Position Limits
        upperPosLim = [14, 0.3, 14, 14, 14]
        lowerPosLim = [-14, -16, -14, -14, -14]
        innfos.poslimit(self.actuID, upperPosLim, lowerPosLim)
        time.sleep(0.5)
    
    def getCurrentPos(self):
        currentPos = innfos.readpos(self.actuID)
        return currentPos
    
    def setRest(self, pos = []):
        if pos:
            self.RestPos = pos
        else:
            currentPos = self.getCurrentPos()
            self.RestPos = currentPos
    
    def isMoving(self):
        initalPos = self.getCurrentPos()
        finalPos = self.getCurrentPos()
        if functools.reduce(lambda x, y : x and y, map(lambda p, q: abs(p-q) < self.posError,initalPos,finalPos), True): 
            return False
        else: 
            return True
    
    def waitUntilStoped(self):
        while self.isMoving():
            time.sleep(0.5)
            
    def powerUp(self, mode):
        print("Powering On")
        if mode == 'fancy':
            innfos.setpos(self.actuID, self.FancyPos)
            self.waitUntilStoped()
        innfos.setpos(self.actuID, self.HomePos)
        self.waitUntilStoped()
    
    def powerDown(self):
        print("Powering Off")
        self.waitUntilStoped()
        innfos.setpos(self.actuID, self.RestPos)
        self.waitUntilStoped()
        innfos.disableact(self.actuID)
    
    def checkConnection(self):
        try:
            statusg = innfos.handshake()
            data = innfos.queryID(5)
            innfos.enableact(data)
            innfos.disableact(data)
            if statusg ==1:
                print('Connection is ok')
            else:
                print('Connection failed')
                sys.exit()
        except Exception as e:
            print('Connection Error:', e)
            sys.exit()
    
    def stopRobot(self):
        while self.isMoving():
            stopAt = self.getCurrentPos()
            self.moveTo(stopAt)
    
    def updateMovementParams(self, newVal, mode, motorNum = None):
        """
        Parameters
        ----------
        newVal : list or a number; The value or values of the parameter (mode) to update.
        mode : the parameter to change
        motorNum : If you are editing only one motor, supply the motor number
        -------

        """
        # Update all the Motors
        if motorNum == None:
            # Make Sure that the user Inputed the Correct Type
            if type(newVal) != list or len(newVal) != len(self.maxSpeed):
                print("Please Provide a List of Speeds for All Actuators")
                return None
            
            # Update the Correct Movement Parameter
            if mode == 'speed':                
                # If You are Slowing it Down, Stop the Robot First
                if self.maxSpeed[0] > newVal[0]:
                    # Change the Value
                    self.maxSpeed = newVal
                    self.stopRobot()
                    self.click_Move_Arm()
                else:
                    # Change the Value
                    self.maxSpeed = newVal
            elif mode == "accel":
                self.accel = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.maxSpeed[0] > newVal[0]:
                    self.stopRobot()
                    self.click_Move_Arm()
            elif mode == "decel":
                self.decel = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.maxSpeed[0] > newVal[0]:
                    self.stopRobot()
                    self.click_Move_Arm()
            else:
                print("No Parameter was Given; None were Changed")
            
        # Update Only One Motor's Value
        else:
            # Make Sure that the user Inputed the Correct Type
            if type(newVal) == list or motorNum >= len(self.maxSpeed) or motorNum < 0:
                print("Please Provide a Single Number Between 0 and", len(self.maxSpeed)-1)
                return None
            
            # Update the Correct Movement Parameter
            if mode == 'speed':
                self.maxSpeed[motorNum] = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.maxSpeed[0] > newVal[0]:
                    self.stopRobot()
                    self.click_Move_Arm()
            elif mode == "accel":
                self.accel[motorNum] = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.maxSpeed[0] > newVal[0]:
                    self.stopRobot()
                    self.click_Move_Arm()
            elif mode == "decel":
                self.decel[motorNum] = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.maxSpeed[0] > newVal[0]:
                    self.stopRobot()
                    self.click_Move_Arm()
            else:
                print("No Parameter was Given; None were Changed")
                
        # Set the new limits
        innfos.trapposset(self.actuID, self.accel, self.maxSpeed, self.decel)                
            


# In[ ]:


class moveRobot(initiateRobot):
    
    def __init__(self, MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5):
        super().__init__(MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5)
    
    def homePos(self):
        # Start at Home
        innfos.setpos(self.actuID, self.HomePos)
        self.waitUntilStoped()
    
    def moveTo(self, pos):
        # Start at Home
        innfos.setpos(self.actuID, pos)
    
    def askUserForInput(self, mode = "oneTime"):
        currentPos = self.getCurrentPos()
        print(currentPos)
        print("Enter New Coordinates or Enter Y to Keep Current One")
        askUser = True
        while askUser:
            userPos = []
            for i in range(0, 5):
                corPos = input("Enter element No-{}: ".format(i+1))
                print(corPos)
                if corPos == "Y":
                    userPos.append(currentPos[i]) 
                else:
                    userPos.append(float(corPos))
            print("The entered list is: \n",userPos)
            self.moveTo(userPos)
            if mode == "oneTime":
                askUser = False
            else:
                keepGoing = input("Type Y to Keep Going")
                if keepGoing != "Y":
                    askUser = False
    
    def moveLeft(self):
        currentPos = self.getCurrentPos()
        currentPos[0] -= 1
        innfos.setpos(self.actuID, currentPos)
    
    def moveRight(self):
        currentPos = self.getCurrentPos()
        currentPos[0] += 1
        innfos.setpos(self.actuID, currentPos)
    
    def moveUp(self):
        currentPos = self.getCurrentPos()
        errorPos = 0.001
        if currentPos[2] < 6 + errorPos:
            currentPos[3] -=1
            currentPos[1] +=1
        else:
            currentPos[2] -= 1
            currentPos[3] -= 1
        innfos.setpos(self.actuID, currentPos)
    
    def moveDown(self):
        currentPos = self.getCurrentPos()
        if currentPos[1] > -14 + self.posError:
            if abs(currentPos[1] - self.HomePos[1]) < self.posError:
                currentPos[3] = currentPos[3] -8
            else:
                currentPos[3] +=1
            currentPos[1] -=1
        else:
            currentPos[2] += 1
            currentPos[3] += 1
        innfos.setpos(self.actuID, currentPos)

# In[ ]:
    

def find_hand():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if p.serial_number == '7593231313935131D162':
            port = p.device         
    return port

port = find_hand()
hand = serial.Serial(port, baudrate=115200, timeout=1)
_ = hand.read_all()



# In[ ]:


'''function design'''

def finger_1(pos,speed=1):
    com_f = 'h1'
    if int(speed) > 5:
        speed = 5
    elif int(speed) <=0:
        speed =0
    if len(str(pos)) == 1:
        pos = '00' + str(pos)
    elif len(str(pos)) == 2:
        pos = '0' + str(pos)
    elif len(str(pos)) == 3:
        pos = str(pos)
    com = com_f + str(pos) + str(speed)
    hand.write(str.encode(com))

def finger_2(pos,speed=1):
    com_f = 'h2'
    if int(speed) > 5:
        speed = 5
    elif int(speed) <=0:
        speed =0
    if len(str(pos)) == 1:
        pos = '00' + str(pos)
    elif len(str(pos)) == 2:
        pos = '0' + str(pos)
    elif len(str(pos)) == 3:
        pos = str(pos)
    com = com_f + str(pos) + str(speed)
    hand.write(str.encode(com))
    
def finger_3(pos,speed=1):
    com_f = 'h3'
    if int(speed) > 5:
        speed = 5
    elif int(speed) <=0:
        speed =0
    if len(str(pos)) == 1:
        pos = '00' + str(pos)
    elif len(str(pos)) == 2:
        pos = '0' + str(pos)
    elif len(str(pos)) == 3:
        pos = str(pos)
    com = com_f + str(pos) + str(speed)
    hand.write(str.encode(com))
    
def finger_4(pos,speed=1):
    com_f = 'h4'
    if int(speed) > 5:
        speed = 5
    elif int(speed) <=0:
        speed =0
    if len(str(pos)) == 1:
        pos = '00' + str(pos)
    elif len(str(pos)) == 2:
        pos = '0' + str(pos)
    elif len(str(pos)) == 3:
        pos = str(pos)
    com = com_f + str(pos) + str(speed)
    hand.write(str.encode(com))
    
def finger_5(pos,speed=1):
    com_f = 'h5'
    if int(speed) > 5:
        speed = 5
    elif int(speed) <=0:
        speed =0
    if len(str(pos)) == 1:
        pos = '00' + str(pos)
    elif len(str(pos)) == 2:
        pos = '0' + str(pos)
    elif len(str(pos)) == 3:
        pos = str(pos)
    com = com_f + str(pos) + str(speed)
    hand.write(str.encode(com))
    
def finger_all(pos,speed=1):
    com_f = 'h6'
    if int(speed) > 5:
        speed = 5
    elif int(speed) <=0:
        speed =0
    if len(str(pos)) == 1:
        pos = '00' + str(pos)
    elif len(str(pos)) == 2:
        pos = '0' + str(pos)
    elif len(str(pos)) == 3:
        pos = str(pos)
    com = com_f + str(pos) + str(speed)
    hand.write(str.encode(com))


# In[ ]:


'''Open the graphic user interface'''
if __name__ == "__main__":# In[ ]:
    # Start GUI
    app = QtWidgets.QApplication(sys.argv)
    # Create GUI
    MainWindow = QtWidgets.QMainWindow()
    centralwidget = QtWidgets.QWidget(MainWindow)
    Number_distance = QtWidgets.QLabel(centralwidget)
    # Set Up Arms
    arm_1 = QtWidgets.QTextEdit(centralwidget)
    arm_2 = QtWidgets.QTextEdit(centralwidget)
    arm_3 = QtWidgets.QTextEdit(centralwidget)
    arm_4 = QtWidgets.QTextEdit(centralwidget)
    arm_5 = QtWidgets.QTextEdit(centralwidget)
    
    # Initiate the GUI and Setup Params
    ui = Ui_MainWindow(MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5, initiate=True)
    
    # Initiate the Robot
    RoboArm = initiateRobot(MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5)
    RoboArm.checkConnection()
    try:
        # Setup the Robot's Parameters
        RoboArm.setRoboParams() # Starts Position Mode. Sets the Position Limits, Speed, and Acceleration  
        RoboArm.setRest()       # Sets the Rest Position to Current Start Position
        
        # Initate Robot for Movement and Place in Beginning Position
        Controller = moveRobot(MainWindow, centralwidget, Number_distance, 
                       arm_1, arm_2, arm_3, arm_4, arm_5)
        
        distanceRead()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
        time.sleep(5)
        # Turn Off the Laser
        hand.write(str.encode('s0')) # turn off the laser
        hand.close()
        # Turn Off Robot
        RoboArm.powerDown() # power off in 5 seconds




