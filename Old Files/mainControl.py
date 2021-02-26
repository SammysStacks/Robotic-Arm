#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:34:49 2021

@author: samuelsolomon
"""

# Import Modules
import sys
# Imort Helper Files
sys.path.append('./Helper Files/')  # Folder with All the Helper Files
import connectArduinoFindPeaks as dataControl # Functions to Stream in Data and Find Peaks
import moveRobot as robotControl              # Functions to Control Robot Movement
import neuralNetwork as NeuralNet             # Machine Learning Code


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    
    # General Data Collection Information
    numDataPoints = 250000  # Number of Points to Stream into the Arduino
    seeFullPlot = True      # Graph the Peak Analysis IN ADDITION TO the Arduino Data
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    saveInputData = True                  # Saves the Data Streamed into From the Arduino as saveExcelName
    saveExcelName = "Recorded Data.xlsx"  # Name of File (Will Overwrite if Already Present)
    saveDataFolder = "Training Data/"     # Data Folder to Save the Arudino Data; MUST END IN '/'
    sheetName = "Trial 1"                 # If SheetName Already Exists, Excel Will Add 1 to the end (The Copy Number) 
    
    # Instead of Arduino Data, Use Test Data from Excel File
    useTestData = True                             # Uses the test Data Provided in testDataExcelFile on Sheet testSheetNum
    testDataExcelFile = "./Input Data/Test Data/channel.xlsx" # Path to the Test Data
    testSheetNum = 0 # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
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
            dataControl.getTestData(testDataExcelFile, seeFullPlot, testSheetNum)
        else:
            dataControl.daq_stream_async(numDataPoints, seeFullPlot)
            
        # Save the Data (if Wanted)
        if saveInputData:
            dataControl.saveTestData(saveDataFolder, saveExcelName, sheetName)
        
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