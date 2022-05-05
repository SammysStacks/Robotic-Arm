# Import Modules
import time
import sys
import linecache
import functools
import threading
# Import innfosControlFunctions.py (Must be in Same Folder!)
import innfosControlFunctions as innfos
# import Read Arduino
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
sys.path.append('../Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataArduino as readArduinoClass   # Functions to Read in Data from Arduino

# --------------------------------------------------------------------------- #
# --------------------------- User Can Edit --------------------------------- #

class initiateRobotArm():
    def __init__(self, guiApp = None):        
        # Label Actuators
        self.actuID = [0x01, 0x02, 0x03, 0x04, 0x05]
        
        # Define Common Positions
        self.HomePos = [0, 5, 8, -5, -3]   # Set the Start/End Home Position
        self.FancyPos = [-1, 6, 8, -5, 0] # Set the Start/End Home Position
        self.RestPos = [-1.2004829049110413, 0.2907487154006958, 1.065185546875, 1.10516357421875, 0.00054931640625] # Set the Start/End Home Position
        self.posError = 0.0007
        
        # Define Action Positions
        self.startGrabTable = [-1, -13, 10, 14, 4]
        self.midGrabTable1 = [-1, -14, 11, 16, 4]
        self.midGrabTable2 = [-1, -15, 12, 18, 4]
        self.endGrabTable = [-1, -16, 12, 19, 4]
        
        # Define Movement Parameters
        self.maxSpeed = [0.2, 0.2, 0.2, 0.4, 0.4]
        self.Speed = [0.1, 0.1, 0.1, 0.3, 0.3]
        self.accel = [1.5, 1.5, 1.5, 1.5, 1.5]
        self.decel = [-1.5, -1.5, -1.5, -1.5, -1.5]
        
        # Specify the Gui
        self.guiApp = guiApp
    
    def setRoboParams(self):
        # Find and Connect to Actuators
        innfos.enableact(self.actuID)
        innfos.trapposmode(self.actuID)
        time.sleep(0.5)
        
        # Set Speed, Acceleration, and Deceleration
        innfos.trapposset(self.actuID, self.accel, self.Speed, self.decel)
        time.sleep(0.5)
        
        # Set Position Limits
        upperPosLim = [20, 10, 20, 40, 40]
        lowerPosLim = [-20, -25, -20, -40, -40]
        innfos.poslimit(self.actuID, upperPosLim, lowerPosLim)
        time.sleep(0.5)
    
    def getCurrentPos(self):
        currentPos = innfos.readpos(self.actuID)
        return currentPos
    
    def setRest(self, pos = []):
        if pos:
            self.RestPos = pos
        else:
            self.waitUntilStoped(waitTime = 1)
            currentPos = self.getCurrentPos()
            self.RestPos = currentPos.copy()
    
    def isMoving(self, sampleTime = 0.3):
        initalPos = self.getCurrentPos()
        time.sleep(sampleTime)
        finalPos = self.getCurrentPos()
        if functools.reduce(lambda x, y : x and y, map(lambda p, q: abs(p-q) < self.posError,initalPos,finalPos), True): 
            return False
        else: 
            return True
    
    def waitUntilStoped(self, waitTime = 0.5):
        while self.isMoving():
            time.sleep(waitTime)
            
    def powerUp(self, startMode, fancyStart = False):
        innfos.trapposset(self.actuID, self.accel, self.maxSpeed, self.decel)
        print("\nPowering On the Robot\n")
        # Add a Fancy Starting Position
        if fancyStart:
            innfos.setpos(self.actuID, self.FancyPos)
        self.waitUntilStoped()
        print("Move to Home")
        # Power Up Into Starting Position
        innfos.setpos(self.actuID, self.HomePos)
        self.waitUntilStoped(waitTime = 0.5)
        innfos.trapposset(self.actuID, self.accel, self.Speed, self.decel)
    
    def powerDown(self, setRest = True):
        print("Powering Off the Robot")
        self.waitUntilStoped(waitTime = 2)
        if setRest:
            innfos.setpos(self.actuID, self.RestPos)
            self.waitUntilStoped(waitTime = 2)
        innfos.disableact(self.actuID)
    
    def checkConnection(self):
        try:
            statusg = innfos.handshake()
            data = innfos.queryID(5)
            innfos.enableact(data)
            innfos.disableact(data)
            if statusg ==1:
                print('Connected to the Robot')
            else:
                print('Connection to the Robot Failed')
                sys.exit()
        except Exception as e:
            print('Connection Error:', e)
            sys.exit()
    
    def stopRobot(self):
        while self.isMoving(sampleTime = 0.125):
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
            if type(newVal) != list or len(newVal) != len(self.Speed):
                print("Please Provide a List of Speeds for All Actuators")
                return None
            
            # Update the Correct Movement Parameter
            if mode == 'speed':                
                # If You are Slowing it Down, Stop the Robot First
                if self.Speed[0] > newVal[0]:
                    # Change the Value
                    self.Speed = newVal
                    #self.stopRobot()
                    if self.guiApp:
                       self.guiApp.click_Move_Arm()
                else:
                    # Change the Value
                    self.Speed = newVal
            elif mode == "accel":
                self.accel = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.Speed[0] > newVal[0]:
                    #self.stopRobot()
                    if self.guiApp:
                        self.guiApp.click_Move_Arm()
            elif mode == "decel":
                self.decel = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.Speed[0] > newVal[0]:
                    #self.stopRobot()
                    if self.guiApp:
                        self.guiApp.click_Move_Arm()
            else:
                print("No Parameter was Given; None were Changed")
            
        # Update Only One Motor's Value
        else:
            # Make Sure that the user Inputed the Correct Type
            if type(newVal) == list or motorNum >= len(self.Speed) or motorNum < 0:
                print("Please Provide a Single Number Between 0 and", len(self.Speed)-1)
                return None
            
            # Update the Correct Movement Parameter
            if mode == 'speed':
                self.Speed[motorNum] = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.Speed[0] > newVal[0]:
                    #self.stopRobot()
                    if self.guiApp:
                        self.guiApp.click_Move_Arm()
            elif mode == "accel":
                self.accel[motorNum] = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.Speed[0] > newVal[0]:
                    #self.stopRobot()
                    if self.guiApp:
                        self.guiApp.click_Move_Arm()
            elif mode == "decel":
                self.decel[motorNum] = newVal
                # If You are Slowing it Down, Stop the Robot First
                if self.Speed[0] > newVal[0]:
                    #self.stopRobot()
                    if self.guiApp:
                        self.guiApp.click_Move_Arm()
            else:
                print("No Parameter was Given; None were Changed")
                
        # Set the new limits
        innfos.trapposset(self.actuID, self.accel, self.Speed, self.decel)                


class moveArm(initiateRobotArm):
    
    def __init__(self, guiApp = None):
        #super(type(self), self).__init__()
        super().__init__(guiApp)
    
    def goHome(self):
        # Start at Home
        innfos.setpos(self.actuID, self.HomePos)
        self.waitUntilStoped()
    
    def moveTo(self, pos, waitFirst = False, waitLast = False, waitTime = 0.5):
        if waitFirst:
            self.waitUntilStoped(waitTime)
        # Move Robot
        innfos.setpos(self.actuID, pos)
        if waitLast:
            self.waitUntilStoped(waitTime)
    
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
            currentPos[3] +=1
            currentPos[1] -=1
        else:
            currentPos[2] += 1
            currentPos[3] += 1
        innfos.setpos(self.actuID, currentPos)
        

class moveHand():
    def __init__(self, handArduino):
        # Hand Port
        self.handArduino = handArduino
        # Specify Hand Parameters
        self.numFingers = 5
    
    def grabHand(self, grabAngle = "45"):
        self.moveFinger(grabAngle, com_f = "h6", speed = 1)
    
    def releaseHand(self, releaseAngle = "90"):
        self.moveFinger(releaseAngle, com_f = "h6", speed = 1)
    
    def resetFingers(self):
        print("Resetting")
        for fingerInd in range(self.numFingers):
            time.sleep(2)
            self.setFingerPos(fingerIndex = fingerInd+1, pos = "90")
        
    def moveFinger(self, pos, com_f = 'h1', speed = 1):
        if int(speed) > 5:
            speed = 5
        elif int(speed) <=0:
            speed = 0
        if len(pos) == 1:
            arduinoPos = '00' + pos
        elif len(pos) == 2:
            arduinoPos = '0' + pos
        elif len(pos) == 3:
            arduinoPos = pos
        com = com_f + str(arduinoPos) + str(speed)
        # Move the Finger
        self.handArduino.write(str.encode(com))
        # Update UI
        self.updateFingerText(int(com_f[-1]), pos)

class robotControl(moveArm, moveHand):
    
    def __init__(self, handArduino = None, guiApp = None):
        # Initiate Parent Classes
        moveHand.__init__(self, handArduino)
        moveArm.__init__(self, guiApp)
    
    def userAction(self):
        # Move Down to Table 
        self.moveTo(self.startGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.midGrabTable1, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.midGrabTable2, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.endGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
        # Grab the Object
        self.grabHand(grabAngle = "40")
        time.sleep(1)
        # Move Back Up to Start
        self.moveTo(self.midGrabTable2, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.midGrabTable1, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.startGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
        # Move Down to Table 
        self.moveTo(self.startGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.midGrabTable1, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.midGrabTable2, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.endGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
        # Release the Object
        self.releaseHand(releaseAngle = "90")
        time.sleep(1)
        # Move Back Up to Start
        self.moveTo(self.midGrabTable2, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.midGrabTable1, waitFirst = True, waitLast = False, waitTime = 0.5)
        self.moveTo(self.startGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
            
    def updateFingerText(self, fingerIndex, fingerPos):
        if self.guiApp:
            if fingerIndex < 6:
                self.guiApp.fingerText[fingerIndex - 1].setText(fingerPos)
            else:
                for fingerInd in range(5):
                    self.guiApp.fingerText[fingerInd].setText(fingerPos)

# --------------------------------------------------------------------------- #
# ------------------------- Defined Program --------------------------------- #

if __name__ == "__main__":
    # General Data Collection Information (You Will Likely Not Edit These)
    handSerialNum = '7593231313935131D162'  # Hand Arduino's Serial Number (port.serial_number)
    emgSerialNum = '85735313333351E040A0'   # EMG Arduino's Serial Number (port.serial_number)
    n_data = 10000
    
    # Get Hand and EMG Arudinos
    readArduino = readArduinoClass.arduinoRead(emgSerialNum, handSerialNum)

    # Initiate the Robot
    robotControl = robotControl(handArduino = readArduino.handArduino, guiApp = None)
    robotControl.checkConnection()
    
    try:
        # Setup the Robot's Parameters
        robotControl.setRoboParams()  # Starts Position Mode. Sets the Position Limits, Speed, and Acceleration  
        robotControl.setRest()        # Sets the Rest Position to Current Start Position            
        
        # Initate Robot for Movement and Place in Beginning Position
        robotControl.powerUp("", fancyStart = False) # If mode = 'fancy', begin there. Then go to Home Position

        # ----------------- User Defined Robotic Movement ------------------- #
        # Set Up Laser Reading
        threading.Thread(target = readArduino.distanceRead, args = (robotControl, n_data), daemon=True).start()
        # Turn Laser On
        readArduino.handArduino.write(str.encode('s1'))

        # Move Down to Table 
        robotControl.moveTo(robotControl.startGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.midGrabTable1, waitFirst = False, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.midGrabTable2, waitFirst = False, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.endGrabTable, waitFirst = False, waitLast = True, waitTime = 1)
        # Grab the Object
        robotControl.grabHand(grabAngle = "45"); time.sleep(2)
        # Turn Laser Off
        readArduino.handArduino.write(str.encode('s0'))
        readArduino.killDistanceRead = True
        robotControl.updateMovementParams([0.1, 0.1, 0.1, 0.3, 0.3], 'speed')
        # Move Back Up to Start
        robotControl.moveTo(robotControl.midGrabTable2, waitFirst = True, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.midGrabTable1, waitFirst = False, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.startGrabTable, waitFirst = False, waitLast = True, waitTime = 1)
        # Move Down to Table 
        robotControl.moveTo(robotControl.startGrabTable, waitFirst = True, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.midGrabTable1, waitFirst = False, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.midGrabTable2, waitFirst = False, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.endGrabTable, waitFirst = False, waitLast = True, waitTime = 1)
        # Release the Object
        robotControl.releaseHand(releaseAngle = "90"); time.sleep(2)
        # Move Back Up to Start
        robotControl.moveTo(robotControl.midGrabTable2, waitFirst = True, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.midGrabTable1, waitFirst = False, waitLast = False, waitTime = 0.5)
        robotControl.moveTo(robotControl.startGrabTable, waitFirst = False, waitLast = True, waitTime = 1)
        # ------------------------------------------------------------------- #
        
    # If Something Goes Wrong, Power Down Robot (Controlled)
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        print(e)
        time.sleep(5)    
        # Turn Off Robot
        robotControl.powerDown(setRest = True)
        readArduino.emgArduino.close()
        readArduino.handArduino.close()
    
    # Power Down Robot
    print("Finished Moving")
    time.sleep(2)
    robotControl.powerDown(setRest = True)
    readArduino.emgArduino.close()
    readArduino.handArduino.close()
    
