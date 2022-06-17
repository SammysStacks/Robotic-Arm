"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Data Aquisition:
    
    All channels were placed on the right arm
    Channel 1 (bottom channel of the arduino used in this experiment) represents
    the EMG signal from the muscle located on top of the arm (with the hand facing 
    palm down). For clarity, this is the muscle that moves the most when the wrist
    is flexed up (with the palm of the hand facing resting on the table)
    Channel 2 (second lowest on arduino) is to the right of channel one on the arm
    with respect to the user looking directly at the hand (away from thumb)
    
    Each Channel Consists of 3 Electrodes: Two EMG Electrodes + 1 EMG Reference
    The Standard Setup Consists of Placing the Electrodes along a muscle group.
    The Reference Electrode Should be Placed in the Middle, And the Electrodes
    Should Line Up On the Axis From the hand to the Elbow (If Using Lower Arm).
    Provide Decent Spacing Between the Electrodes (Noticeable Gap)
    
    Robotic Control Note: 
        
    The input values for the robotic arm MAY be different for you.
    If you run this code and the arm moves to  bad position, please edit
    './Helper Files/Robotic Control/moveRobot.py' file. In particular, you
    should edit the first class's (initiateRobotArm) init method. Specifically,
    the homePos and FancyPos attributes. They are an array of 5 numbers corresponding
    to the actuators on the Innfos robot. Innfos has already changed their name to
    Minta and may make further edits to the robot in the future.
    
    HardWare Processing:
    The Code Below Used the Following Electronic Material from Olimex:  
        Circuit Board: https://www.olimex.com/Products/Duino/Shields/SHIELD-EKG-EMG/open-source-hardware
        Electrodes: https://www.olimex.com/Products/Duino/Shields/SHIELD-EKG-EMG-PRO/
    
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        $ conda install scikit-learn
        $ conda install matplotlib
        $ conda install tensorflow
        $ conda install openpyxl
        $ conda install pyserial
        $ conda install joblib
        $ conda install numpy
        $ conda install keras
        $ conda install shap
        
    --------------------------------------------------------------------------
"""
# --------------------------------------------------------------------------- #

# Basic Modules
import sys
import time
import linecache
import threading
import numpy as np
from pathlib import Path

# Import GUI Template
sys.path.append('./Helper Files/GUI Design/')  # Folder with GUI Files
import GUI as GUI   # Function with GUI and Finger Movements

# Imort Robotic Control Files
sys.path.append('./Helper Files/Robotic Control/')  # Folder with All the Helper Files
import moveRobot as robotController              # Functions to Control Robot Movement

# Import Data Aquisition and Analysis Files
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import emgAnalysis as emgAnalysis         # Functions to Analyze the EMG Data
import readDataExcel as excelData         # Functions to Save/Read in Data from Excel
import readDataArduino as streamData      # Functions to Read in Data from Arduino

# Import Machine Learning Files
sys.path.append('./Helper Files/Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    
    # General Data Collection Information (You Will Likely Not Edit These)
    emgSerialNum = '85735313333351E040A0'    # Arduino Serial Number (port.serial_number) Collecting EMG Signals
    handSerialNum = '7593231313935131D162'   # Arduino Serial Number for the Robotic Hand Control. Leave None if NOT Controlling the Hand
    numDataPoints = 50000   # The Number of Points to Stream into the Arduino
    numTimePoints = 3000    # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 200    # The Number of NEW Data Points to Analyze at a Time; My Beta-Test Used 200 Points with Plotting (100 Without). This CAN Change How SOME Peaks are Found (be Careful)
    samplingFreq = 800      # The Average Number of Points Steamed Into the Arduino Per Second; If NONE Given, Algorithm will Calculate Based on Initial Data
    numChannels = 4         # The Number of Arduino Channels with EMG Signals Read in; My Beta-Test Used 4 Channels
    numFeatures = 4         # The Number of Features to Extract/Save/Train on
    # Specify the Type of Movements to Learn
    gestureClasses = np.char.lower(["Up", "Down", "Left", "Right", "Grab", "Release"])  # Define Labels as Array
    
    # Protocol Switches: Only One Can be True; Only the First True Variable Excecutes
    streamArduinoData = False  # Stream in Data from the Arduino and Analyze; Input 'testModel' = True to Apply Learning
    useRoboticGUI = True       # Do not stream in data and control the robot using the GUI
    trainModel = False         # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = False   # Graph the Data to Show Incoming Signals + Analysis
    saveModel = False          # Save the Machine Learning Model for Later Use
    saveData = False           # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName' or map2D if Training
    
    # ---------------------------------------------------------------------- #
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveData:
        saveExcelName = "You Yu 11-12-2021 Finger Angles.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Output Data/EMG Data/Upper Back/2021-11-11/"  # Data Folder to Save the Excel Data; MUST END IN '/'
        # Speficy the Movement You Will Perform
        currentMovement = "Full Grab".lower() # Make Sure it is Lowercase
        if currentMovement and currentMovement not in gestureClasses:
            print("The Gesture", "'" + currentMovement + "'", "is Not in", gestureClasses)
    else:
        saveDataFolder = None
    
    if useRoboticGUI:
        controlTimeSeconds = 60*100
    
    # Specify Training Location
    if trainModel:
        trainDataExcelFolder = "../Input Data/Full Training Data/Lab Electrodes/Sam/May11/"  # Path to the Training Data Folder; All .xlsx Data Used
    
    if streamArduinoData or trainModel:
        # Pick the Machine Learning Module to Use
        modelType = "KNN"  # Machine Learning Options: NN, RF, LR, KNN, SVM
        modelPath = "./Machine Learning Modules/Models/predictionModelKNNFull_SamArm1.pkl" # Path to Model (Creates New if it Doesn't Exist)
        # Get the Machine Learning Module
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, dataDim = numChannels, gestureClasses = gestureClasses, saveDataFolder = saveDataFolder)
        predictionModel = performMachineLearning.predictionModel

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
            robotControl = robotController.robotControl(handArduino = None, guiApp = guiApp)
            robotControl.checkConnection()
            
            # Setup the Robot's Parameters and Initialize Home Position
            robotControl.setRoboParams()  # Starts Position Mode. Sets the Position Limits, Speed, and Acceleration  
            robotControl.setRest()        # Sets the Rest Position to Current Start Position            
            robotControl.powerUp("", fancyStart = True) # If mode = 'fancy', begin there. Then go to Home Position
            
            # Stream in EMG Arduino Data and Perform Gesture Recognition
            arduinoRead = streamData.arduinoRead(emgSerialNum = emgSerialNum, handSerialNum = handSerialNum)
            readData = streamData.emgArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData, guiApp = guiApp)
            threading.Thread(target = readData.streamEMGData, args = (numDataPoints, predictionModel, robotControl), daemon=True).start()
            
            # Start UI Popup
            guiApp.app.exec_()
            
            # Power Down the Robot
            robotControl.powerDown(setRest = False)
            guiApp.resetButton()
            
            # Save the Data Streamed in (if Wanted)
            if saveData:
                # Format Sheet Name
                sheetName = "Trial 1 - "  # If SheetName Already Exists, Increase Trial # by One
                sheetName = sheetName + currentMovement
                # Double Check to See if User Wants to Save the Data
                verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
                if verifiedSave.upper() == "Y":
                    # Initialize Class to Save the Data and Save
                    saveInputs = excelData.saveExcel(numChannels, numFeatures)
                    saveInputs.saveData(readData.data, readData.featureList, saveDataFolder, saveExcelName, sheetName, currentMovement)
                else:
                    print("User Chose Not to Save the Data")
        # ---------------------- Only Use Robotic GUI ----------------------- #
        if useRoboticGUI:
            # Initiate the UI
            guiApp = GUI.Ui_MainWindow(handArduino = None)
        
            # Initiate the Robotic Control
            robotControl = robotController.robotControl(handArduino = None, guiApp = guiApp)
            robotControl.checkConnection()
            
            # Setup the Robot's Parameters and Initialize Home Position
            robotControl.setRoboParams()  # Starts Position Mode. Sets the Position Limits, Speed, and Acceleration  
            robotControl.setRest()        # Sets the Rest Position to Current Start Position            
            robotControl.powerUp("", fancyStart = True) # If mode = 'fancy', begin there. Then go to Home Position
            
            # Stream in EMG Arduino Data and Perform Gesture Recognition
            arduinoRead = streamData.arduinoRead(emgSerialNum = None, handSerialNum = handSerialNum)
            readData = streamData.emgArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData, guiApp = guiApp)
            threading.Thread(target = readData.controlRobotManually, args = (numDataPoints, controlTimeSeconds, robotControl), daemon=True).start()
            
            # Start UI Popup
            guiApp.app.exec_()
            
            # Power Down the Robot
            robotControl.powerDown(setRest = False)
            guiApp.resetButton()
        # ------------------------- Train ML Model -------------------------- #
        elif trainModel:
            # Create Portocol
            emgProtocol = emgAnalysis.emgProtocol(numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData)
            # Extract the Data
            readData = excelData.readExcel(emgProtocol)
            signalData, signalLabels = readData.getTrainingData(trainDataExcelFolder, numFeatures, gestureClasses, mode='Train')
            print("\nCollected Signal Data")
            # Train the Data on the Gestures
            performMachineLearning.trainModel(signalData, signalLabels)
            # Save Signals and Labels
            if saveData and performMachineLearning.map2D:
                saveInputs = excelData.saveExcel(numChannels, numFeatures)
                saveExcelNameMap = Path(saveExcelName).stem + "_mapedData.xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
                saveInputs.saveLabeledPoints(performMachineLearning.map2D, signalLabels,  performMachineLearning.predictionModel.predictData(signalData), saveDataFolder, saveExcelNameMap, sheetName = "Signal Data and Labels")
            # Save the Neural Network (The Weights of Each Edge)
            if saveModel:
                 performMachineLearning.predictionModel.saveModel(modelPath)
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
    
