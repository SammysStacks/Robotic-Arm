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
        $ conda install -c conda-forge tensorflow
        % conda install -c conda-forge ffmpeg
        $ conda install scikit-learn
        $ conda install matplotlib
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
    numDataPoints = 50000   # The Number of Points to Stream into the Arduino
    numTimePoints = 10000    # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 200    # The Number of NEW Data Points to Analyze at a Time; My Beta-Test Used 200 Points with Plotting (100 Without). This CAN Change How SOME Peaks are Found (be Careful)
    samplingFreq = None     # The Average Number of Points Steamed Into the Arduino Per Second; If NONE Given, Algorithm will Calculate Based on Initial Data
    numChannels = 4         # The Number of Arduino Channels with EMG Signals Read in; My Beta-Test Used 4 Channels
    # Specify the Type of Movements to Learn
    gestureClasses = np.char.lower(["Up", "Down", "Left", "Right", "Grab", "Release"])  # Define Labels as Array
    #gestureClasses = np.char.lower(["Forwards", "Back", "Left", "Right", "Rotate Left", "Rotate Right"])  # Define Labels as Array
    #gestureClasses = np.char.lower(["Front 90", "Back", "Left", "Right 90", "Open Chest", "Shrug"])  # Define Labels as Array
    #gestureClasses = np.char.lower(["Front 45", "Front 90", "Front 135", "Front 180"])  # Define Labels as Array
    #gestureClasses = np.char.lower(["Fingers Curl", "Fingers to Thumb", "Fingers to Palm", "Full Grab"])  # Define Labels as Array
    #gestureClasses = np.char.lower(["Back", "Left", "Open Chest", "Shrug", "Right 45", "Right 90", "Right 135", "Right 180", "Front 45", "Front 90", "Front 135", "Front 180"])  # Define Labels as Array

    # Protocol Switches: Only One Can be True; Only the First True Variable Excecutes
    streamArduinoData = False  # Stream in Data from the Arduino and Analyze; Input 'testModel' = True to Apply Learning
    readDataFromExcel = True   # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    reAnalyzePeaks = False     # Read in ALL Data Under 'trainingDataExcelFolder', and Reanalyze Peaks (THIS EDITS EXCEL DATA IN PLACE!; DONT STOP PROGRAM MIDWAY)
    trainModel = False         # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = False    # Graph the Data to Show Incoming Signals + Analysis
    saveModel = False          # Save the Machine Learning Model for Later Use
    testModel = True          # Apply the Learning Algorithm to Decode the Signals
    saveData = False           # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName' or map2D if Training
    
    # ---------------------------------------------------------------------- #
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveData:
        saveExcelName = "You Yu 11-12-2021 Finger Angles.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Output Data/EMG Data/Upper Back/2021-11-11/"  # Data Folder to Save the Excel Data; MUST END IN '/'
        # Speficy the eye Movement You Will Perform
        eyeMovement = "Full Grab".lower() # Make Sure it is Lowercase
        if eyeMovement and eyeMovement not in gestureClasses:
            print("The Gesture", "'" + eyeMovement + "'", "is Not in", gestureClasses)
    else:
        saveDataFolder = False
            
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        testDataExcelFile = "../Output Data/EMG Data/Arm/May11 Test Results/New Analysis/Samuel Solomon 2021-05-11 Round 1 All 5 Features.xlsx" # Path to the Test Data
        testSheetNum = 5   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Use Previously Processed Data that was Saved; Extract Features for Training
    if reAnalyzePeaks or trainModel:
        trainingDataExcelFolder = "../Output Data/EMG Data/Arm/May11 Test Results/New Analysis/" # Path to the Training Data Folder; All .xlsx Data Used

    if trainModel or testModel and not reAnalyzePeaks:
        # Pick the Machine Learning Module to Use
        modelType = "KNN"  # Machine Learning Options: RF, LR, KNN, SVM
        modelPath = "./Machine Learning/Models/predictionModel_Arm_" + modelType + ".pkl" # Path to Model (Creates New if it Doesn't Exist)
        if trainModel:
            saveDataFolder = trainingDataExcelFolder + "Del/" + modelType + "/"
        else:
            saveDataFolder = None
        # Get the Machine Learning Module
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, dataDim = numChannels, gestureClasses = gestureClasses, saveDataFolder = saveDataFolder)
        predictionModel = performMachineLearning.predictionModel
        # Feature Labels
        featureLabels = []
        # ["peakAverage", "peakHeight", "peakVariance", "peakSTD", "maxSlope"]
        for feature in ["peakAverage", "peakHeight", "peakEnergy", "peakSTD", "maxSlope"]:
            for channel in range(1,5):
                featureLabels.append(feature + " in Channel " + str(channel))
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
        emgProtocol = streamData.emgArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData, guiApp = None)
        emgProtocol.streamEMGData(numDataPoints, predictionModel = predictionModel, actionControl = None)
    # Take Data from Excel Sheet
    elif readDataFromExcel:
        readData = excelData.readExcel(emgProtocol)
        readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = predictionModel, actionControl = None)
    # Redo Peak Analysis
    elif reAnalyzePeaks:
        readData = excelData.readExcel(emgProtocol)
        readData.getTrainingData(trainingDataExcelFolder, numChannels, gestureClasses, mode='reAnalyze')
    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        # Extract the Data
        readData = excelData.readExcel(emgProtocol)
        signalData, signalLabels = readData.getTrainingData(trainingDataExcelFolder, numChannels, gestureClasses, mode='Train')
        print("\nCollected Signal Data")

        # Train the Data on the Gestures
        performMachineLearning.trainModel(signalData, signalLabels, featureLabels = featureLabels)
        # Save Signals and Labels
        if len(performMachineLearning.map2D) != 0:
            saveInputs = excelData.saveExcel(numChannels, numChannels)
            saveExcelNameMap = "MapedData_" + modelType + ".xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
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
            saveInputs = excelData.saveExcel(numChannels, numChannels)
            saveInputs.saveData(emgProtocol.data, emgProtocol.featureList, saveDataFolder, saveExcelName, sheetName, eyeMovement)
        else:
            print("User Chose Not to Save the Data")

        
