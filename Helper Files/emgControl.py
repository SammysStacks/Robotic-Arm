"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Data Aquisition:
    Each Channel Consists of 3 Electrodes: Two EMG Electrodes + 1 EMG Reference
    The Standard Setup Consists of Placing the Electrodes along a muscle group.
    The Reference Electrode Should be Placed in the Middle, And the Electrodes
    Should Line Up On the Axis From the hand to the Elbow (If Using Lower Arm).
    Provide Decent Spacing Between the Electrodes (Noticeable Gap)
    
    HardWare Processing:
    The Code Below Used the Following Electronic Material from Olimex:  
        Circuit Board: https://www.olimex.com/Products/Duino/Shields/SHIELD-EKG-EMG/open-source-hardware
        Electrodes: https://www.olimex.com/Products/Duino/Shields/SHIELD-EKG-EMG-PRO/
    
    --------------------------------------------------------------------------
    
    Note: Not ALL of the Modules are Required for Every Run (Some are ONYL if Plotting)
    Modules to Import Before Running the Program (Some May be Missing):
        % conda install -c conda-forge ffmpeg
        $ conda install scikit-learn
        $ conda install matplotlib
        $ conda install tensorflow
        $ conda install openpyxl
        $ conda install pyserial
        $ conda install joblib
        $ conda install numpy
        $ conda install keras
        
        
    --------------------------------------------------------------------------
"""
# Use '%matplotlib qt' to View Plot

# Basic Modules
import sys
import numpy as np
from pathlib import Path

# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataExcel as excelData         # Functions to Save/Read in Data from Excel
import readDataArduino as streamData      # Functions to Read in Data from Arduino
import emgAnalysis as emgAnalysis         # Functions to Analyze the EMG Data

# Import Files for Machine Learning
sys.path.append('./Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # General Data Collection Information (You Will Likely Not Edit These)
    emgSerialNum = '85735313333351E040A0'    # Arduino Serial Number (port.serial_number) Collecting EMG Signals
    handSerialNum = None   # Arduino Serial Number for the Robotic Hand Control. Leave None if NOT Controlling the Hand
    numDataPoints = 15000   # The Number of Points to Stream into the Arduino
    numTimePoints = 5000    # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 500    # The Number of NEW Data Points to Analyze at a Time; My Beta-Test Used 200 Points with Plotting (100 Without). This CAN Change How SOME Peaks are Found (be Careful)
    samplingFreq = 800     # The Average Number of Points Steamed Into the Arduino Per Second; If NONE Given, Algorithm will Calculate Based on Initial Data
    numChannels = 4         # The Number of Arduino Channels with EMG Signals Read in; My Beta-Test Used 4 Channels
    numFeatures = 4         # The Number of Features to Extract/Save/Train on
    # Specify the Type of Movements to Learn
    #gestureClasses = np.char.lower(["Up", "Down", "Left", "Right", "Grab", "Release"])  # Define Labels as Array
    gestureClasses = np.char.lower(["Forwards", "Back", "Left", "Right", "Rotate Left", "Rotate Right"])  # Define Labels as Array

    # Protocol Switches: Only One Can be True; Only the First True Variable Excecutes
    streamArduinoData = False  # Stream in Data from the Arduino and Analyze; Input 'testModel' = True to Apply Learning
    readDataFromExcel = True   # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    reAnalyzePeaks = False     # Read in ALL Data Under 'trainingDataExcelFolder', and Reanalyze Peaks (THIS EDITS EXCEL DATA IN PLACE!; DONT STOP PROGRAM MIDWAY)
    trainModel = False         # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = True    # Graph the Data to Show Incoming Signals + Analysis
    saveModel = False          # Save the Machine Learning Model for Later Use
    testModel = True          # Apply the Learning Algorithm to Decode the Signals
    saveData = False           # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName' or map2D if Training
    
    # ---------------------------------------------------------------------- #
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveData:
        saveExcelName = "Samuel Solomon 2021-11-05 Neck Test.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Output Data/EMG Data/Neck/"  # Data Folder to Save the Excel Data; MUST END IN '/'
        # Speficy the eye Movement You Will Perform
        eyeMovement = "Back".lower() # Make Sure it is Lowercase
        if eyeMovement not in gestureClasses:
            print("The Gesture", "'" + eyeMovement + "'", "is Not in", gestureClasses)
            
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        testDataExcelFile = "../Input Data/Full Training Data/Industry Electrodes/Samuel Solomon (Pure) 2021-03-17.xlsx" # Path to the Test Data
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Use Previously Processed Data that was Saved; Extract Features for Training
    if reAnalyzePeaks or trainModel:
        trainingDataExcelFolder = "../Output Data/EMG Data/Neck/"  # Path to the Training Data Folder; All .xlsx Data Used

    if trainModel or testModel:
        # Pick the Machine Learning Module to Use
        modelType = "NN"  # Machine Learning Options: NN, RF, LR, KNN, SVM
        modelPath = "./Machine Learning/Models/predictionModelKNNFull_SamArm1.pkl" # Path to Model (Creates New if it Doesn't Exist)
        # Get the Machine Learning Module
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, dataDim = numChannels, gestureClasses = gestureClasses)
        predictionModel = performMachineLearning.predictionModel
    else:
        predictionModel = None

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    if not streamArduinoData:
        emgProtocol = emgAnalysis.emgProtocol(numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData)
    # Stream in Data from Arduino
    if streamArduinoData:
        arduinoRead = streamData.arduinoRead(emgSerialNum = emgSerialNum, handSerialNum = handSerialNum)
        readData = streamData.emgArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData, guiApp = None)
        readData.streamEMGData(numDataPoints, predictionModel = predictionModel, actionControl = None)
    # Take Data from Excel Sheet
    elif readDataFromExcel:
        readData = excelData.readExcel(emgProtocol)
        readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = predictionModel, actionControl = None)
    # Redo Peak Analysis
    elif reAnalyzePeaks:
        readData = excelData.readExcel(emgProtocol)
        readData.getTrainingData(trainingDataExcelFolder, numFeatures, gestureClasses, mode='reAnalyze')
    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        # Extract the Data
        readData = excelData.readExcel(emgProtocol)
        signalData, signalLabels = readData.getTrainingData(trainingDataExcelFolder, numFeatures, gestureClasses, mode='Train')
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
        
    
    # Save the Data in Excel: EMG Channels (Cols 1-4); X-Peaks (Cols 5-8); Peak Features (Cols 9-12)
    if saveData and not trainModel and not reAnalyzePeaks:
        # Format Sheet Name
        sheetName = "Trial 1 - "  # If SheetName Already Exists, Increase Trial # by One
        sheetName = sheetName + eyeMovement
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        if verifiedSave.upper() == "Y":
            # Initialize Class to Save the Data and Save
            saveInputs = excelData.saveExcel(numChannels, numFeatures)
            saveInputs.saveData(readData.data, readData.featureList, saveDataFolder, saveExcelName, sheetName, eyeMovement)
        else:
            print("User Chose Not to Save the Data")

        
