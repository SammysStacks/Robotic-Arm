#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:34:49 2021

@author: samuelsolomon
"""

# Import Modules
import sys
import numpy as np
import os
import openpyxl as xl
# Imort Helper Files
sys.path.append('./Helper Files/')  # Folder with All the Helper Files
import connectArduinoFindPeaks as dataControl # Functions to Stream in Data and Find Peaks
import moveRobot as robotControl              # Functions to Control Robot Movement

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    
   # General Data Collection Information
    numDataPoints = 5000  # Number of Points to Stream into the Arduino
    seeFullPlot = True    # Graph the Peak Analysis IN ADDITION TO the Arduino Data
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    saveInputData = False                  # Saves the Data Streamed into From the Arduino as saveExcelName
    saveExcelName = "Samuel Solomon.xlsx" # Name of File (Will Overwrite if Already Present)
    saveDataFolder = "./Training Data/"  # Data Folder to Save the Arudino Data; MUST END IN '/'
    handMovement = "Right" # Speficy the hand Movement if Streaming in Data
    
    # Instead of Arduino Data, Use Test Data from Excel File
    useTestData = False                  # Uses the test Data Provided in testDataExcelFile on Sheet testSheetNum
    testDataExcelFile = "./Input Data/Test Data/channel.xlsx" # Path to the Test Data
    testSheetNum = 0                     # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Using and Saving the Neural network
    useTrainingData = True
    trainDataExcelFolder = "./Training Data/"
    
    trainNeuralNetwork = True
    testNeuralNetwork = False
    SaveNeuralNetwork = False
    saveNeuralNetworkName = "testNet1"
    saveNeuralNetworkFolder = "./Neural Network/"
    
    # Variables Users Can Change, BUT SHOULDNT
    sheetName = "Trial 1 - "  # If SheetName Already Exists, Excel Will Add 1 to the end (The Copy Number) 
    sheetName = sheetName + handMovement
    
    numChannels = 4
    
    movementOptions = np.array(["Right", "Left", "Up", "Down", "Grab", "Release"])
    movementOptions = np.char.lower(movementOptions)
    
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #                EMG Program (Should Not Have to Edit)                   #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    # Initiate the Robot
    RoboArm = robotControl.initiateRobot()
    RoboArm.checkConnection()
    try:
        # Setup the Robot's Parameters
        RoboArm.setRoboParams() # Starts Position Mode. Sets the Position Limits, Speed, and Acceleration  
        RoboArm.setRest()       # Sets the Rest Position to Current Start Position
        
        # Initate Robot for Movement and Place in Beginning Position
        Controller = robotControl.moveRobot()
        Controller.powerUp('fancy') # If mode = 'fancy', begin there. Then go to Home Position
        
        # Begin Data Collection
        if useTestData:
            dataControl.getTestData(testDataExcelFile, seeFullPlot, testNeuralNetwork, testSheetNum, Controller=Controller)
        elif useTrainingData:
            NN_Data = np.empty((0, numChannels), float)
            NN_Labels = np.empty((0, len(movementOptions)), float)
            for excelFile in list(os.listdir(trainDataExcelFolder)):
                if excelFile.endswith(".xlsx") and not excelFile.startswith("~"):
                    # Get Full Path to the Excel File
                    trainingExcelFile = trainDataExcelFolder + excelFile
                    print("\nLoading Excel File", trainingExcelFile)
                    # Load the Excel File
                    WB = xl.load_workbook(trainingExcelFile, data_only=True, read_only=True)
                    WB_worksheets = WB.worksheets
                    # Loop Over Each Sheet in the File
                    for excelSheet in WB_worksheets:
                        # Get the Training Data/Label from the Sheet
                        NN_Data, NN_Labels = dataControl.getTrainingData(excelSheet, NN_Data, NN_Labels, movementOptions)
            NaN_Placements = np.isnan(NN_Data)
            NN_Data[NaN_Placements] = 0
            print(NN_Data, "\n\n", NN_Labels)
        else:
            dataControl.daq_stream_async(numDataPoints, seeFullPlot, testNeuralNetwork, Controller=Controller)
            
        # Save the Data (if Wanted)
        if saveInputData:
            dataControl.saveTestData(saveDataFolder, saveExcelName, handMovement, sheetName)
        
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
    
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #