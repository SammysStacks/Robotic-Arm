# Import Modules
import time
import sys
import functools
# Import innfosControlFunctions.py (Must be in Same Folder!)
import innfosControlFunctions as innfos

# --------------------------------------------------------------------------- #
# --------------------------- User Can Edit --------------------------------- #

class initiateRobot:
    def __init__(self):
        # Label Actuators
        self.actuID = [0x01, 0x02, 0x03, 0x04, 0x05]
        
        # Define Common Positions
        self.HomePos = [-1, -10, 5, 12, 0] # Set the Start/End Home Position
        self.FancyPos = [-1, -14, -8, 13, 0] # Set the Start/End Home Position
        self.RestPos = [-1.2004829049110413, 0.2907487154006958, 1.065185546875, 1.10516357421875, 0.00054931640625] # Set the Start/End Home Position
        self.posError = 0.01
        
        # Define Movement Parameters
        self.maxSpeed = [0.1, 0.1, 0.1, 0.1, 0.1]
        self.accel = [1.5, 1.5, 1.5, 1.5, 1.5]
        self.decel = [-0.75, -0.75, -0.75, -0.75, -0.75]
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
        time.sleep(0.5)
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
                self.maxSpeed = newVal
            elif mode == "accel":
                self.accel = newVal
            elif mode == "decel":
                self.decel = newVal
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
            elif mode == "accel":
                self.accel[motorNum] = newVal
            elif mode == "decel":
                self.decel[motorNum] = newVal
            else:
                print("No Parameter was Given; None were Changed")
                
        # Set the new limits
        innfos.trapposset(self.actuID, self.accel, self.maxSpeed, self.decel)                
            
        

# --------------------------------------------------------------------------- #
# ---------------------------- Move Robot ----------------------------------- #

class moveRobot(initiateRobot):
    
    def __init__(self):
        super().__init__()
    
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
        self.waitUntilStoped()
    
    def moveRight(self):
        currentPos = self.getCurrentPos()
        currentPos[0] += 1
        innfos.setpos(self.actuID, currentPos)
        self.waitUntilStoped()
    
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
        self.waitUntilStoped()
    
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
        self.waitUntilStoped()


# --------------------------------------------------------------------------- #
# ------------------------- Defined Program --------------------------------- #

if __name__ == "__main__":
    # Initiate the Robot
    RoboArm = initiateRobot()
    RoboArm.checkConnection()
    try:
        # Setup the Robot's Parameters
        RoboArm.setRoboParams() # Starts Position Mode. Sets the Position Limits, Speed, and Acceleration  
        RoboArm.setRest()       # Sets the Rest Position to Current Start Position
        
        # Initate Robot for Movement and Place in Beginning Position
        Controller = moveRobot()
        Controller.powerUp('fancy') # If mode = 'fancy', begin there. Then go to Home Position
        
        # User Defined Movements
        Controller.moveLeft()
        Controller.moveRight()
        Controller.moveDown()
        Controller.moveUp()
        
    # If Something Goes Wrong, Power Down Robot (Controlled)
    except:
        RoboArm.powerDown()
    
    # Power Down Robot
    RoboArm.powerDown()
    
