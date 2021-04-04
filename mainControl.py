"""
    Written by Samuel Solomon and Jiahong Li
    
    --------------------------------------------------------------------------
    Data Aquisition:
    
    All channels were placed on the right arm
    Channel 1 (bottom channel of the arduino used in this experiment) represents
    the EMG signal from the muscle located on top of the arm (with the hand facing 
    palm down). For clarity, this is the muscle that moves the most when the wrist
    is flexed up (with the palm of the hand facing resting on the table)
    Channel 2 (second lowest on arduino) is to the right of channel one on the arm
    with respect to the user looking directly at the hand (away from thumb)
    
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        $ conda install -c conda-forge ffmpeg
        $ conda install matplotlib
        $ conda install tensorflow
        $ conda install openpyxl
        $ conda install sklearn
        $ conda install joblib
        $ conda install numpy
        $ conda install keras
        
    --------------------------------------------------------------------------
"""


"""
UI Needs to Move the Robot and the Robot Needs to Change the UI: CIRCULAR
Sampling Frequency of Laser (currently 0.76; need 0.05)
UI is NOT asychronous and the program wont run after the UI is called
"""

# --------------------------------------------------------------------------- #

# Basic Modules
import sys
import time
import numpy as np
import collections
import linecache
import threading

# Neural Network Modules
from sklearn.model_selection import train_test_split

# Import GUI Template
sys.path.append('./Helper Files/GUI Design/')  # Folder with GUI Files
import GUI as GUI   # Function with GUI and Finger Movements

# Imort Robotic Control Files
sys.path.append('./Helper Files/Robotic Control/')  # Folder with All the Helper Files
import moveRobot as robotController              # Functions to Control Robot Movement

# Import Data Aquisition and Analysis Files
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataArduino as streamData    # Functions to Read in Data from Arduino
import readDataExcel as excelData       # Functions to Save/Read in Data from Excel

# Import Machine Learning Files
sys.path.append('./Helper Files/Machine Learning Modules/')  # Folder with Machine Learning Files
import KNN as KNN   # Functions for K-Nearest Neighbors' Algorithm

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # General Data Collection Information (You Will Likely Not Edit These)
    handSerialNum = '7593231313935131D162'  # Hand Arduino's Serial Number (port.serial_number)
    emgSerialNum = '85735313333351E040A0'   # EMG Arduino's Serial Number (port.serial_number)
    numDataPoints = 20000  # The Number of Points to Stream into the EMG Arduino
    moveDataFinger = 200   # The Number of Data Points to Analyze at a Time
    numChannels = 4        # The Number of EMG Arduino Channels with EMG Signals Read
    xWidth = 1000          # The Number of Data Points to Display to the User (Plot) at a Time
    
    # Protocol Switches: Only One Can be True; Only the First True Variable Excecutes
    streamArduinoData = True  # Stream in Data from the Arduino and Analyze
    trainModel = False         # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Option During the Run
    saveInputData = False # Saves the Data Streamed in as 'saveExcelName'
    seeFullPlot = True    # Graph the Peak Analysis IN ADDITION TO the Arduino Data
    SaveModel = False     # Save the Machine Learning Model for Later Use
        
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveInputData:
        saveExcelName = "Samuel Solomon 2021-03-18.xlsx"  # The Name of the Saved File
        saveDataFolder = "./Input Data/Full Training Data/"   # Data Folder to Save the Excel Data; MUST END IN '/'
        handMovement = "Release"                          # Speficy the hand Movement You Will Perform
        
        sheetName = "Trial 1 - "  # If SheetName Already Exists, Excel 1 to Trial #
        sheetName = sheetName + handMovement
    
    # Define Labels as Array
    movementOptions = np.array(["Right", "Left", "Up", "Down", "Grab", "Release"])
    movementOptions = np.char.lower(movementOptions)
    # Edge Case: User Defines a Movement that Does not Exist, Return Error
    if saveInputData and handMovement.lower() not in movementOptions:
        print("\nUser Defined an Unknown Hand Gesture")
        print("The Gesture", "'" + handMovement.lower() + "'", "is Not in", movementOptions)
        sys.exit()
        
    # Specify the ML Module
    modelPath = "./Helper Files/Machine Learning Modules/Models/myModelKNNFull.pkl"
    MLModel = KNN.KNN(modelPath = modelPath, numClasses = len(movementOptions))    
    
    trainDataExcelFolder = "./Input Data/Full Training Data/"  # Path to the Training Data Folder; All .xlsx Data Used

    # ---------------------------------------------------------------------- #
    #                EMG Program (Should Not Have to Edit)                   #
    # ---------------------------------------------------------------------- #
        
    try:
        # ----------------- Stream EMG Data and Move Robot ------------------ #
        # Begin Data Collection and Analysis (Move Robot During Movements)
        if streamArduinoData:
            # Initiate the UI
            guiApp = GUI.Ui_MainWindow(handArduino = None)
        
            # Initiate the Robotic Control
            robotControl = robotController.robotControl(guiApp)
            robotControl.checkConnection()
            
            # Setup the Robot's Parameters and Initialize Home Position
            robotControl.setRoboParams()  # Starts Position Mode. Sets the Position Limits, Speed, and Acceleration  
            robotControl.setRest()        # Sets the Rest Position to Current Start Position            
            robotControl.powerUp('fancy') # If mode = 'fancy', begin there. Then go to Home Position
    
            # Stream in EMG Arduino Data and Perform Gesture Recognition
            readData = streamData.arduinoRead(emgSerialNum, handSerialNum, xWidth, moveDataFinger, numChannels, movementOptions, guiApp)
            threading.Thread(target = readData.streamArduinoData, args = (numDataPoints, seeFullPlot, MLModel, robotControl), daemon=True).start()
            
            # Start UI Popup
            guiApp.app.exec_()
            
            # Power Down the Robot
            robotControl.powerDown()
            
            # Save the Data Streamed in (if Wanted)
            if saveInputData:
                saveInputs = excelData.saveExcel(numChannels)
                saveInputs.saveData(readData.data, readData.xTopGrouping, readData.featureSetGrouping, saveDataFolder, saveExcelName, sheetName, handMovement)
        # ------------------------- Train ML Model -------------------------- #
        elif trainModel:
            # Read in Training Data/Labels
            readData = excelData.readExcel(xWidth, moveDataFinger, numChannels, movementOptions)
            signalData, signalLabels = readData.getTrainingData(trainDataExcelFolder, movementOptions, mode='Train')
            print("\nCollected Signal Data")
            
            # Split the Data Randomly into Training and Testing Data Sets
            signalLabelsClass = [np.argmax(i) for i in signalLabels]
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabelsClass, test_size=0.1, shuffle= True, stratify=signalLabels)
        
            # Train the Classifier (the Model) with the Training Data
            MLModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
            # Plot the Training Loss    
            #MLModel.plotModel(signalData, signalLabelsClass)
            
            # Find the Class Data Distribution in the Total Training/Testing Set
            classDistribution = collections.Counter(signalLabelsClass)
            print("Class Distribution:", classDistribution)
            print("Number of Data Points = ", len(classDistribution))

            # Save the Classifier: if Desired
            if SaveModel:
                MLModel.saveModel(modelPath)
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
        robotControl.powerDown()
    
    

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
