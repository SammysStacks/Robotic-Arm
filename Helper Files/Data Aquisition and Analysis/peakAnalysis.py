# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:17:05 2021
    conda install -c conda-forge ffmpeg

@author: Samuel Solomon
"""

# Basic Modules
import os
import math
import sys
import numpy as np

# Peak Detection
import scipy
import scipy.signal
from scipy.signal import lfilter
# Plotting
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class globalParam:
    
    def __init__(self, xWidth = 2000, moveDataFinger = 200, numChannels = 4, movementOptions = []):
        
        # Input Parameters
        self.numChannels = numChannels        # Number of EMG Signals
        self.xWidth = xWidth                  # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger  # The Amount of Data to Stream in Before Finding Peaks
        self.movementOptions = movementOptions
        
        # Data to Stream in
        self.data = {}
        
        # Peak Finding and Feature Holders
        self.RMSDataList = {}
        self.mapPeakLocToYPos = {}
        self.xTopGrouping = {}
        self.featureSetGrouping = {}
        
        # Start with Fresh Inputs
        self.resetGlobalVariables()
        
        # Define Class for Plotting Peaks
        self.initPlotPeaks()

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = dict(time_ms=[])
        for channel in range(self.numChannels):
            self.data['Channel'+str(1+channel)] = []
        
        # Peak Finding and Feature Holders
        for channelNum in range(self.numChannels):
            # Hold Analysis Values
            self.RMSDataList[channelNum] = []
            self.mapPeakLocToYPos[channelNum] = {}
            self.xTopGrouping[channelNum] = {1:[]}
            self.featureSetGrouping[channelNum] = {1:[]}


    def initPlotPeaks(self): 

        # Specify Figure Asthetics
        self.peakCurrentRightColorOrder = {
            0: "tab:red",
            1: "tab:purple",
            2: "tab:orange",
            3: "tab:pink",
            4: "tab:brown",
            5: "tab:green",
            6: "tab:gray",
            7: "tab:cyan",
            }
        
        # use ggplot style for more sophisticated visuals
        #plt.style.use('ggplot')



        # ------------------------------------------------------------------- #
        # --------- Plot Variables user Can Edit (Global Variables) --------- #
        # High Pass Filter Parameters
        self.f1 = 100; self.f3 = 50;
        self.Rp = 0.1; self.Rs = 30;
        self.samplingFreq = 1000
        # Root Mean Squared (RMS) Parameters
        self.rmsWindow = 250; self.stepSize = 8;
        
        # Data Collection Parameters
        self.peakDetectionBufferSize = 500
        self.featureDefinitionBuffer = 50
        self.maxPeakSep = 100   # Seperation that Defines a New Group
        self.badInitialHighPass = max(self.rmsWindow + self.stepSize, 5000)  # Must be > rmsWindow + stepSize; Current Experimental lowest: xWidth*0.4 (Changes with xWidth)
        
            
        # Specify Figure aesthetics
        plt.ion()
        figWidth = 14; figHeight = 10; # 28,16
        self.fig, ax = plt.subplots(4, 2, sharey=False, sharex = 'col', figsize=(figWidth, figHeight))
        
        # Plot the Raw Data
        self.xWidth = 2000
        self.xWidthPeaks = 10000
        self.xLimLow = 0; self.xLimHigh = self.xLimLow + self.xWidth;
        self.yLimLow = 0; self.yLimHigh = 5; 
        self.movieGraphChannelListRaw = []
        self.channelListRaw = []
        for channelNum in range(self.numChannels):
            # Create Plots
            self.channelListRaw.append(ax[channelNum, 0])
            
            # Generate Plot
            self.movieGraphChannelListRaw.append(self.channelListRaw[channelNum].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            
            # Set Figure Limits
            self.channelListRaw[channelNum].set_xlim(self.xLimLow, self.xLimHigh)
            self.channelListRaw[channelNum].set_ylim(self.yLimLow, self.yLimHigh)
            # Label Axis + Add Title
            self.channelListRaw[channelNum].set_title("EMG Signal in Channel " + str(channelNum + 1))
            self.channelListRaw[channelNum].set_xlabel("EMG Data Points")
            self.channelListRaw[channelNum].set_ylabel("EMG Signal (Volts)")
            
        # Create the Peak Data Plot
        self.yLimitHighFiltered = 0.5;
        self.channelListFiltered = [] 
        # Plot the Peak Data
        self.movieGraphChannelListFiltered = []
        self.movieGraphChannelTopPeaksList = []
        self.movieGraphChannelBottomPeaksList = []
        for channelNum in range(self.numChannels):
            # Create Plots
            self.channelListFiltered.append(ax[channelNum, 1])
            
            # Plot RMS Peaks
            self.movieGraphChannelListFiltered.append(self.channelListFiltered[channelNum].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            # Plot Top Peaks
            self.movieGraphChannelTopPeaksList.append({})
            
            # Set Figure Limits
            self.channelListFiltered[channelNum].set_ylim(self.yLimLow, self.yLimitHighFiltered)
            # Label Axis + Add Title
            self.channelListFiltered[channelNum].set_title("Filtered EMG Signal in Channel " + str(channelNum + 1))
            self.channelListFiltered[channelNum].set_xlabel("Root Mean Squared Data Point")
            self.channelListFiltered[channelNum].set_ylabel("Filtered Signal (Volts)")
            
            # Hold Analysis Values
            self.RMSDataList[channelNum] = []
            self.mapPeakLocToYPos[channelNum] = {}
            self.xTopGrouping[channelNum] = {1:[]}
            self.featureSetGrouping[channelNum] = {1:[]}
            
        
        # Tighten Figure White Space (Must be After wW Add Fig Info)
        self.fig.tight_layout(pad=2.0)
        plt.show()
        
        # Create Output File Directory to Save Data
        self.outputData = "../Output Data/"
        os.makedirs(self.outputData, exist_ok=True)
        
        # Keep Track of Robotic Movements
        self.lastPeakAnalyzed = 0
        self.currentGroupNum = -1
        
        self.xDataEMG = self.data['time_ms'][0:self.xWidth]

    
    def live_plotter(self, dataFinger, seeFullPlot = False, myModel = None, Controller=None):
                
        # Get X Data: Shared Axis for All Channels
        self.xDataEMG = self.data['time_ms'][dataFinger:dataFinger + self.xWidth]
        # Add incoming Data to Each Respective Channel's Plot
        for dataChannel in range(self.numChannels):
            
            # ---------------------- Get EMG Signal -----------------------------#
            # Get New Y DataData
            newYData = self.data['Channel' + str(dataChannel+1)][dataFinger:dataFinger + self.xWidth]
            # Plot Raw EMG Data            
            self.movieGraphChannelListRaw[dataChannel].set_data(self.xDataEMG, newYData)
            self.channelListRaw[dataChannel].set_xlim(self.xDataEMG[0], self.xDataEMG[0]+self.xWidth)
            # -------------------------------------------------------------------#
            
            # ---------------------- High pass Filter ---------------------------#
            # Calculate the Number of New Data Points You Need for Peak Detecting
            RMSData = self.RMSDataList[dataChannel]
            if len(RMSData) == 0 and dataFinger > 0:
                print("You Collected NO RMS Data Last Round ... ?")
                print("Please Decrease Your rmsWindow Size or Increase xWidth")
                sys.exit()
            else:
                newRMSDataBegins = self.stepSize*(len(RMSData)) - (dataFinger)
    
            # High Pass Filter to Remove Noise
            startHPF = max(dataFinger + newRMSDataBegins - self.badInitialHighPass,0)
            yDataBuffer = self.data['Channel' + str(dataChannel+1)][startHPF:dataFinger + self.xWidth]
            filteredData = self.highPassFilter(yDataBuffer, self.f1, self.f3, self.Rp, self.Rs, self.samplingFreq)[-self.xWidth + newRMSDataBegins:]   
            # -------------------------------------------------------------------#
    
            # --------------------- Root Mean Squared ---------------------------#
            if len(filteredData) >= self.rmsWindow:
                newRMSData = self.RMSFilter(filteredData, self.rmsWindow, self.stepSize)
                RMSData.extend(newRMSData)
            else:
                print("If You are Here, it is a Mistake")
                print("Please Decrease Your rmsWindow Size or Increase xWidth")
                sys.exit()
            xDataRMS = self.data['time_ms'][max(len(RMSData) - self.xWidthPeaks,0):len(RMSData)]  
            # Get New Peak if you Enough Points have Passed
            self.currentGroupNum = max(self.xTopGrouping[dataChannel].keys(), default=0)
            currentHighestXPeak = max([max(self.xTopGrouping[i][self.currentGroupNum], default=0) for i in range(self.numChannels)])
            if abs(xDataRMS[-1] - currentHighestXPeak) > self.maxPeakSep and currentHighestXPeak != 0:
                self.createNewGroup(myModel, Controller)
            # Plot RMS Data
            self.movieGraphChannelListFiltered[dataChannel].set_data(xDataRMS, RMSData[-self.xWidthPeaks:])
            self.channelListFiltered[dataChannel].set_xlim(xDataRMS[0], xDataRMS[0] + self.xWidthPeaks)
            # -------------------------------------------------------------------#

            # ----------------------- Peak Detection  ---------------------------#
            # Get Most Current RMS Data (Add Buffer in Case the peak is Cut Off)
            bufferRMSData = RMSData[-(len(newRMSData) + self.peakDetectionBufferSize):]
            bufferRMSDataX = xDataRMS[-(len(newRMSData) + self.peakDetectionBufferSize):]
            # Find Peaks from the New Data
            newTopPeaks, self.mapPeakLocToYPos, yBase = self.find_peaks(bufferRMSDataX, bufferRMSData, self.mapPeakLocToYPos, dataChannel, RMSData)
            # If No New Peaks, Then No New Features
            if newTopPeaks == {}:
                continue
            # Split the Peaks into the X,Y Points
            xPeakTop, yPeakTop = zip(*newTopPeaks.items())
            # -------------------------------------------------------------------#
            
            # --------------------- Feature Extraction  -------------------------#
            # If New Peak Was Found with Enough Peak Seperation, Add Group 
            if abs(xPeakTop[0] - currentHighestXPeak) > self.maxPeakSep and currentHighestXPeak != 0:
                self.createNewGroup(myModel, Controller)
            # Features Analysis to Group Peaks Together 
            batchXGroups, featureSetTemp = self.featureDefinition(RMSData, xPeakTop, yBase, self.currentGroupNum, myModel, Controller)
            # Update Overall Grouping Dictionary
            for groupNum in batchXGroups.keys():
                # Get New Peaks/Features to Add
                updateXGroups = batchXGroups[groupNum]
                updateFeatures = featureSetTemp[groupNum]
                # Add Them
                if groupNum in self.xTopGrouping[dataChannel].keys():
                    self.xTopGrouping[dataChannel][groupNum].extend(updateXGroups)
                    self.featureSetGrouping[dataChannel][groupNum].extend(updateFeatures)
                else:
                    self.xTopGrouping[dataChannel][groupNum] = updateXGroups
                    self.featureSetGrouping[dataChannel][groupNum] = updateFeatures
                            
            
            # Plot the Peaks; Colored by Grouping
            yFinger = 0
            yPeakTop = list(self.mapPeakLocToYPos[dataChannel].values())
            for groupNum in self.xTopGrouping[dataChannel].keys():
                # Check to See if the Group Has a Plot You Can Use
                groupPeakPlot = self.movieGraphChannelTopPeaksList[dataChannel].get(groupNum, None)
                # If None Availible, Create a New Plot to Add the Data
                if groupPeakPlot == None:
                    channelFiltered = self.channelListFiltered[dataChannel]
                    # Color Code the Group Peaks. Wrap Around to First Index When Done
                    groupColor = (groupNum-1)%(len(self.peakCurrentRightColorOrder))
                    # Create a Plot for the Peaks Using its Respective Group's Color
                    groupPeakPlot = channelFiltered.plot([], [], 'o', c=self.peakCurrentRightColorOrder[groupColor], linewidth=1, alpha = 0.65)[0]
                    # Save the Plot for Later Use in the Group
                    self.movieGraphChannelTopPeaksList[dataChannel][groupNum] = groupPeakPlot
                # Get Peak Points
                xTopGroup = np.array(self.xTopGrouping[dataChannel][groupNum])
                yTopGroup = yPeakTop[yFinger:yFinger+len(xTopGroup)]
                # Add the Data to the Plot
                xTopGroupPlotWindow = xTopGroup[xTopGroup > xDataRMS[0]]
                yTopGroupPlotWindow = yTopGroup[-len(xTopGroupPlotWindow):]
                # Plot the Peaks in the Group
                groupPeakPlot.set_data(xTopGroupPlotWindow, yTopGroupPlotWindow)
                # Keep Track of Current Y Data with Respect to the X Group Data
                yFinger += len(xTopGroup)
                        
        # Update to Get New Data Next Round
        plt.draw()
        self.fig.canvas.flush_events()
    

    def analyzeFullBatch(self, channelNum = 1):
        print("Printing Seperate test plots")
        # Get Data to Plot
        xData = self.data['time_ms']
        yData = self.data['Channel' + str(channelNum)]
        
        # Get Data and Filter
        plt.figure()
        plt.plot(xData,yData, c='tab:blue', alpha=0.7)
        plt.title("EMG Data")
        
        plt.figure()
        filteredData = self.highPassFilter(yData, self.f1, self.f3, self.Rp, self.Rs, self.samplingFreq)
        plt.plot(xData,filteredData, c='tab:blue', alpha=0.7)
        plt.title("Filtered Data")
        
        plt.figure()
        RMSData = self.RMSFilter(filteredData, self.window, self.step)
        plt.plot(xData[0:len(RMSData)],RMSData, c='tab:blue', alpha=0.7)
        plt.title("RMS Data")
        
        # Find Peaks
        batchTopPeaks = self.find_peaks(xData, RMSData)
        xPeakTop, yPeakTop, yBase = zip(*batchTopPeaks.items())
        xTopGrouping, featureSet = self.featureDefinition(RMSData, xPeakTop, yBase, 0)
    

        


# --------------------------------------------------------------------------- #
# ------------------------- Signal Analysis --------------------------------- #


    def highPassFilter(self, inputData, f1, f3, Rp, Rs, samplingFreq):
        """
        data: Data to Filter
        f1: cutOffFreqPassThrough
        f3: cutOffFreqBand
        Rp: attDB (0.1)
        Rs: cutOffDB (30)
        samplingFreq: Frequecy You Take Data
        """
        Wp = 2*math.pi*f1/samplingFreq
        Ws = 2*math.pi*f3/samplingFreq
        [n, wn] = scipy.signal.cheb1ord(Wp/math.pi, Ws/math.pi, Rp, Rs)
        [bz1, az1] = scipy.signal.cheby1(n, Rp, Wp/math.pi, 'High')
        filteredData = lfilter(bz1, az1, inputData)
        return filteredData
    
    def RMSFilter(self, inputData, rmsWindow=250, stepSize=8):
        """
        The Function loops through the given EMG Data, looking at batches of data
            of size rmsWindow at every interval seperated by stepSize.
        In Each Window, we take the magnitude of the data vector (sqrt[a^2+b^2]
            for [a,b] data point)
        A list of each root mean squared value is returned (in order)
        
        The Final List has a length of 1 + math.floor((len(inputData) - rmsWindow) / stepSize)
        --------------------------------------------------------------------------
        Input Variable Definitions:
            inputData: A List containing the  EMG Data
            rmsWindow: The Amount of Data in the Groups we Analyze via RMS
            stepSize: The Distance Between Data Groups
        --------------------------------------------------------------------------
        """
        normalization = math.sqrt(rmsWindow)
        # Take Root Mean Squared of Batch Data (numBatch = rmsWindow)
        numSteps = 1 + math.floor((len(inputData) - rmsWindow) / stepSize)
        #print("RMS:", numSteps, len(inputData))
        RMSData = np.zeros(numSteps)
        for i in range(numSteps):
            # Get Data in the Window to take RMS
            inputWindow = inputData[i*stepSize:i*stepSize + rmsWindow]
            # Take RMS
            RMSData[i] = np.linalg.norm(inputWindow, ord=2)/normalization
        
        return RMSData

    def featureDefinition(self, RMSData, xTop, yBase, currentGroupNum, myModel = None, Controller = None):
        featureSet = []
        correctionTerm = 0
        # For Every Peak, Take the Average of the Points in Front of it
        for peakNum in range(len(xTop)):
            xTopLoc = xTop[peakNum]
            yBaseline = yBase[peakNum]
            # Get Peak's Points
            featureWindow = RMSData[xTopLoc - self.featureDefinitionBuffer:xTopLoc + self.featureDefinitionBuffer]
            # Take the Average of the Peak as the Feature
            if len(featureWindow) > 0:
                featureSet.append(np.mean(featureWindow) - yBaseline)
            # Edge Effect: np.mean([]) = NaN -> Ignore This Peak Until Next Round
            else:
                correctionTerm += 1
        
        # If No Features/Peaks This Round, Return Empty Dictionaries
        if featureSet == []:
            return {}, {}
            
        # Group the Feature Sets Peaks into One EMG Signal/Group
        peakSeperation = np.diff(xTop[0:len(xTop) - correctionTerm]) # Identify New Signal by Peak Seperation
        peakGrouping = {currentGroupNum:[featureSet[0]]}  # Holder for the Features
        xGrouping = {currentGroupNum:[xTop[0]]}        # Holder for the Corresponding Peaks
        for i, peakSep in enumerate(peakSeperation):
            # A Part of Previous Group
            if peakSep < self.maxPeakSep:
                peakGrouping[currentGroupNum].append(featureSet[i+1])
                xGrouping[currentGroupNum].append(xTop[i+1])
            # New Group
            else:
                self.createNewGroup(myModel, Controller)
                peakGrouping[currentGroupNum] = [featureSet[i+1]]
                xGrouping[currentGroupNum] = [xTop[i+1]]
     
        return xGrouping, peakGrouping


    def find_peaks(self, xData, yData, oldPeaks, channel, RMSData):
        # Convert to Numpy (For Faster Data Processing)
        numpyDataX = np.array(xData)
        numpyDataY = np.array(yData)
        # Find Peak Indices
        peakInfo = scipy.signal.find_peaks(yData, prominence=.03, height=0.01, width=15, rel_height=0.5, distance = 100)
        indicesTop = peakInfo[0]
        # Get X,Y Peaks
        xTop = numpyDataX[indicesTop]
        yTop = numpyDataY[indicesTop]
        
        #yBases = numpyDataY[peakInfo[1]['left_bases']]
        yBases = []
        for top in yTop:
            yBases.append(min(numpyDataY[top-200:top]))
        #print(peakInfo)
        
        # Find the New Peaks
        newTopPeaks = {}; yBase = []
        for i, xLoc in enumerate(xTop):
            if xLoc not in oldPeaks[channel] and xLoc + self.featureDefinitionBuffer > len(RMSData):
                # Record New Peaks and Add New Peaks to Ongoing list
                newTopPeaks[xLoc] = yTop[i]
                oldPeaks[channel][xLoc] = yTop[i]
                yBase.append(yBases[i])
        # Return New Peaks and Update Peak Dictionary
        return newTopPeaks, oldPeaks, yBase
    
    
    def removePeakBackground(self, xTop, yTop, RMSData):
        newY = []
        for pointNum in range(len(xTop)):
            baselineIndex = self.findLeftMinimum(RMSData, xTop[pointNum])
            yPoint = yTop[pointNum] - RMSData[baselineIndex]
            newY.append(yPoint)
        return newY
    
    def createNewGroup(self, myModel, Controller):
        if myModel:
            self.predictMovement(myModel, Controller)
        self.currentGroupNum += 1
        for channel in range(self.numChannels):
            self.xTopGrouping[channel][self.currentGroupNum] = []
            self.featureSetGrouping[channel][self.currentGroupNum] = []
    
    def predictMovement(self, myModel, Controller = None):
        # Get FeatureSet Point for Group
        groupFeatures = []
        for channel in range(self.numChannels):
            # Get the Features for the Group and Take the First One
            channelFeature = self.featureSetGrouping[channel][self.currentGroupNum]
            groupFeatures.append((channelFeature or [0])[0])
        inputData = np.array([groupFeatures])
        if len(inputData[inputData > 0]) <= 1 and np.sum(inputData) <= 0.05:
            print("Only One Small Signal Found; Not Moving Robot")
            return None
        
        # Predict Data
        predictedIndex = myModel.predictData(inputData)[0]
        predictedLabel = self.movementOptions[predictedIndex]
        print("The Predicted Label is", predictedLabel)
        if Controller:
            if predictedLabel == "left":
                Controller.moveLeft()
            elif predictedLabel == "right":
                Controller.moveRight()
            elif predictedLabel == "down":
                Controller.moveDown()
            elif predictedLabel == "up":
                Controller.moveUp()
            elif predictedLabel == "grab":
                Controller.grabHand()
            elif predictedLabel == "release":
                Controller.releaseHand()



 