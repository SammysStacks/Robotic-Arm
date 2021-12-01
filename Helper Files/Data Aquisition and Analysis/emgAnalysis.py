# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:17:05 2021
    conda install -c conda-forge ffmpeg

@author: Samuel Solomon
"""

# Basic Modules
import sys
import math
import numpy as np
# Peak Detection
import scipy
import scipy.signal
from  itertools import chain
# High/Low Pass + Smoothen Filters
from scipy.signal import lfilter
# Plotting
import matplotlib
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class emgProtocol:
    
    def __init__(self, numTimePoints = 2000, moveDataFinger = 200, numChannels = 4, samplingFreq = 800, gestureClasses = [], plotStreamedData = False):
        
        # Input Parameters
        self.numChannels = numChannels        # Number of EMG Signals
        self.numTimePoints = numTimePoints                  # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger  # The Amount of Data to Stream in Before Finding Peaks
        self.gestureClasses = gestureClasses
        self.plotStreamedData = plotStreamedData
        # Data Holders
        self.previousDataRMS = {}
        
        # High Pass Filter Parameters
        self.f1 = 100; self.f3 = 50;
        self.samplingFreq = samplingFreq
        self.Rp = 0.1; self.Rs = 30;
        # Root Mean Squared (RMS) Parameters
        self.rmsWindow = 400; self.stepSize = 10;  # self.rmsWindow = 400; self.stepSize = 10;
        
        # Data Collection Parameters
        self.highPassBuffer = max(self.rmsWindow + self.stepSize, 5000)  # Must be > rmsWindow + stepSize
        self.peakDetectionBuffer = 2000  # Buffer in Case Peaks are Only Half Formed at Edges
        self.numPointsRMS = 5*max(1 + math.floor((numTimePoints - self.rmsWindow) / self.stepSize), 0)         # Number of Root Mean Squared Data (After HPF) to Plot
        
        # Parameters That Rely on Sampling Frequency
        if self.samplingFreq:
            self.initParamsFromSamplingRate()
        
        # Check to See if Parameters Make Sense
        self.checkParams()
        
        # Start with Fresh Inputs
        self.resetGlobalVariables()
        
        # Define Class for Plotting Peaks
        if plotStreamedData:
            # Initialize Plots
            matplotlib.use('Qt5Agg') # Set Plotting GUI Backend            
            self.initPlotPeaks()    # Create the Plots
            
    def initParamsFromSamplingRate(self):
        self.Wp = 2*math.pi*self.f1/self.samplingFreq
        self.Ws = 2*math.pi*self.f3/self.samplingFreq
        # Calculate Number of Streamed Points Per RMS Point
        self.secondsPerPointRMS = self.stepSize/self.samplingFreq   # Seconds per Delta RMS Point

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = {'timePoints':[]}
        for channelIndex in range(self.numChannels):
            self.data['Channel'+str(1+channelIndex)] = []
            # Hold Analysis Values
            self.previousDataRMS[channelIndex] = []
        
        # Reset Mutable Variables
        self.xDataRMS = []              # Holder for Most Recent RMS Data
        self.groupWidthRMS = None       # The Number of Seconds for 1 Group
        self.groupWidthRMSPoints = None # The Number of Points for 1 Group
        self.lastAnalyzedGroup = -1     # Last Fully Formed Group (Peaks in Gesture)
        self.highestAnalyzedGroupStartX = 0  # First X-Point of Last Fully Formed Group 
        # Reset Mutable Lists
        self.xPeaksList = [[] for _ in range(self.numChannels)]
        self.yPeaksList = [[] for _ in range(self.numChannels)]
        self.badPeakInd = [[] for _ in range(self.numChannels)]
        self.featureList = [[] for _ in range(self.numChannels)]
        self.timeDelayIndices = [[] for _ in range(self.numChannels)]
                
        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close()
    
    def checkParams(self):
        if self.moveDataFinger > self.numTimePoints:
            print("You are Analyzing Too Much Data in a Batch. 'moveDataFinger' MUST be Less than 'numTimePoints'")
            sys.exit()
        elif self.rmsWindow < self.stepSize:
            print("'stepSize' Should NOT be Greater Than 'rmsWindow'. This Means You are JUMPING OVER Datapoints (Missing Point).")
            sys.exit()


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
        plt.style.use('seaborn-poster') #sets the size of the charts
        #plt.style.use('ggplot')

        # ------------------------------------------------------------------- #
        # --------- Plot Variables user Can Edit (Global Variables) --------- #

        # Specify Figure aesthetics
        figWidth = 14; figHeight = 10;
        self.fig, axes = plt.subplots(self.numChannels, 2, sharey=False, sharex = 'col', figsize=(figWidth, figHeight))
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 5; 
        self.timeDelayPlotsRaw = []
        self.bioelectricDataPlots = []; self.bioelectricPlotAxes = []
        for channelIndex in range(self.numChannels):
            # Create Plots
            self.bioelectricPlotAxes.append(axes[channelIndex, 0])
            
            # Plot Biolectic Signals
            self.bioelectricDataPlots.append(self.bioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            self.timeDelayPlotsRaw.append(self.bioelectricPlotAxes[channelIndex].plot([], [], '-', c="blue", linewidth=2, alpha = 0.65)[0])

            # Set Figure Limits
            self.bioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            # Label Axis + Add Title
            self.bioelectricPlotAxes[channelIndex].set_title("Bioelectric Signal in Channel " + str(channelIndex + 1))
            self.bioelectricPlotAxes[channelIndex].set_xlabel("Time (Seconds)", fontsize=10)
            self.bioelectricPlotAxes[channelIndex].set_ylabel("Bioelectric Signal (Volts)")
            
        yLimitHighFiltered = 0.5;
        # Create the Data Plots
        self.filteredBioelectricPlotAxes = [] 
        self.filteredBioelectricDataPlots = []
        self.timeDelayPlotsRMS = []
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            # Plot Flitered Peaks
            self.timeDelayPlotsRMS.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="blue", linewidth=2, alpha = 0.65)[0])
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])

            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimitHighFiltered)
            # Label Axis + Add Title
            self.filteredBioelectricPlotAxes[channelIndex].set_title("Filtered Bioelectric Signal in Channel " + str(channelIndex + 1))
            self.filteredBioelectricPlotAxes[channelIndex].set_xlabel("Root Mean Squared Data Point")
            self.filteredBioelectricPlotAxes[channelIndex].set_ylabel("Filtered Signal (Volts)", fontsize=10)
        
        self.filteredBioelectricPeakPlots = [[] for _ in range(self.numChannels)]

        # Tighten Figure White Space (Must be After wW Add Fig Info)
        self.fig.tight_layout(pad=2.0);
        self.fig.canvas.draw()
        
    
    def analyzeData(self, dataFinger, plotStreamedData = False, predictionModel = None, actionControl=None):  
        
        xPeaksHolder = []; yPeaksHolder = []; featureHolder = []; baselineHolder = []
        # Get X Data: Shared Axis for All Channels
        self.timePoints = self.data['timePoints'][dataFinger:dataFinger + self.numTimePoints]
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ---------------------- High pass Filter ----------------------- #
            # Find New Points That Need Filtering
            totalPreviousPointsRMS = max(1 + math.floor((dataFinger + len(self.timePoints) - self.moveDataFinger - self.rmsWindow) / self.stepSize), 0) if dataFinger else 0
            dataPointerRMS = self.stepSize*totalPreviousPointsRMS
            # Add Buffer to New Points as HPF is Bad at Edge
            startHPF = max(dataPointerRMS - self.highPassBuffer, 0)
            
            if not self.samplingFreq:
                # Calculate Sampling Frequency
                self.samplingFreq = len(self.data['timePoints'][startHPF:])/(self.data['timePoints'][-1] - self.data['timePoints'][startHPF])
                print("\tSetting Sampling Frequency to", self.samplingFreq)
                # Parameters That Rely on Sampling Rate
                self.initParamsFromSamplingRate()
            
            # High Pass Filter to Remove Noise
            numNewDataForRMS = dataFinger + len(self.timePoints) - dataPointerRMS
            yDataBuffer = self.data['Channel' + str(channelIndex+1)][startHPF:dataFinger + len(self.timePoints)]
            filteredData = self.highPassFilter(yDataBuffer)[-(numNewDataForRMS):]   
            # --------------------------------------------------------------- #
    
            # --------------------- Root Mean Squared ----------------------- #
            # Calculated the RMS and Add the Data to the Stored Buffer from the Last Round
            oldPointsRMS = len(self.previousDataRMS[channelIndex])
            dataRMS = self.RMSFilter(filteredData, self.previousDataRMS[channelIndex], self.rmsWindow, self.stepSize, dataPointerRMS, channelIndex)[-self.numPointsRMS:]
            if channelIndex == 0:
                self.xDataRMS = self.xDataRMS[-len(dataRMS):]
            
            # Make Sure You are Saving Enough Points for the Next Round
            savePointsRMS = self.peakDetectionBuffer + self.stepSize + self.highPassBuffer + self.numPointsRMS
            self.previousDataRMS[channelIndex] = dataRMS[-savePointsRMS:] # Store RMS Data Needed for Next Round
            # --------------------------------------------------------------- #
            
            # ------------------- Plot Biolectric Signal -------------------- #
            if plotStreamedData:
                # Get New Y Data
                newYData = self.data['Channel' + str(channelIndex+1)][dataFinger:dataFinger + len(self.timePoints)]
                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(self.timePoints, newYData)
                self.bioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1])
                
                # Plot the Filtered (RMS) Data
                self.filteredBioelectricDataPlots[channelIndex].set_data(self.xDataRMS[-self.numPointsRMS:], dataRMS[-self.numPointsRMS:])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(self.xDataRMS[max(0, -self.numPointsRMS)], self.xDataRMS[-1])
            # --------------------------------------------------------------- #

            # ----------------------- Peak Detection ------------------------ #
            # Get Most Current RMS Data (Add Buffer in Case the peak is Cut Off)
            numNewPointsRMS = max(0, len(self.previousDataRMS[channelIndex]) - oldPointsRMS)
            bufferRMSData = dataRMS[-(numNewPointsRMS + self.peakDetectionBuffer):]
            bufferRMSDataX = self.xDataRMS[-(numNewPointsRMS + self.peakDetectionBuffer):]
            # Find Peaks from the New Data
            xPeaksNew, yPeaksNew, peakInds = self.findPeaks(bufferRMSDataX, bufferRMSData, channelIndex)
                        
            # Keep Track of Peaks
            xPeaksHolder.append(xPeaksNew); yPeaksHolder.append(yPeaksNew)
            # --------------------------------------------------------------- #
            
            # ---------------------- Feature Analysis  ---------------------- #
            # Extract Features from the Good Peaks 
            newFeatures, leftBaselines = self.extractFeatures(bufferRMSDataX, bufferRMSData, peakInds)
            
            # Keep Track of the Features
            featureHolder.append(newFeatures); baselineHolder.append(leftBaselines)
            # --------------------------------------------------------------- #


        # ---------------------- Group Peaks Together ----------------------- #
        # Initialize Variables to Group Peaks
        currentGroupInd = len(self.xPeaksList[0]) - 1
        peakPointers = [0 for _ in range(self.numChannels)]
            
        # Identify New Movements by Peak Seperation
        while True:
            nextPeak = None; # Start by Assuming there are NO More Peaks to Analyze
            # Find the Next Peak to be Grouped
            for channelInd in range(self.numChannels):
                peakPointer = peakPointers[channelInd]
                # Check to See if any Peaks are Left in the Channel
                if peakPointer < len(xPeaksHolder[channelInd]):
                    # If the Peak is Smaller, Update the nextPeak
                    if not nextPeak or xPeaksHolder[channelInd][peakPointer] < nextPeak:
                        nextPeak = xPeaksHolder[channelInd][peakPointer]
                        peakChannel = channelInd
            # Update Channel Pointer if Next Peak Exists
            if nextPeak:
                peakPointers[peakChannel] += 1
            # Break out of the Loop if No Next Peak
            else:
                break
                
            # If the Peak is Far from the Last Group, Make a New Group
            if nextPeak - self.highestAnalyzedGroupStartX > self.groupWidthRMS or len(self.xPeaksList[peakChannel]) == 0:
                # Update Group Holders
                for channelInd in range(self.numChannels):
                    self.xPeaksList[channelInd].append([])
                    self.yPeaksList[channelInd].append([])
                    self.featureList[channelInd].append([])
                    self.timeDelayIndices[channelInd].append([])
                # Reset Your New Highest Peak
                self.highestAnalyzedGroupStartX = nextPeak
                currentGroupInd += 1
            elif nextPeak < self.highestAnalyzedGroupStartX:
                continue
                
            # Update Group Holders (Only Add First Peak)
            if self.xPeaksList[peakChannel][currentGroupInd] == []:
                self.xPeaksList[peakChannel][currentGroupInd].append(nextPeak)
                self.yPeaksList[peakChannel][currentGroupInd].append(yPeaksHolder[peakChannel][peakPointers[peakChannel]-1])
                self.featureList[peakChannel][currentGroupInd].append(featureHolder[peakChannel][peakPointers[peakChannel]-1])
                self.timeDelayIndices[peakChannel][currentGroupInd].append(baselineHolder[peakChannel][peakPointers[peakChannel]-1])
        
        # ------------------------ Predict Movement ------------------------- #
        badGroupInds = 0
        # If New Peak Group was Found, Predict the movement                        
        for currentGroupInd in range(self.lastAnalyzedGroup+1, len(self.xPeaksList[0])):
            currentGroupInd -= badGroupInds
            
            # Find the Smallest Peak Added to Unanalyzed Group
            smallestNewPeak = 0; allPeaksCollected = True
            for channelInd in range(self.numChannels):
                channelPeaks = self.xPeaksList[channelInd]
                if channelPeaks and channelPeaks[currentGroupInd]:
                    if channelPeaks[currentGroupInd][0] < smallestNewPeak or not smallestNewPeak:
                        smallestNewPeak = channelPeaks[currentGroupInd][0]
                else:
                    allPeaksCollected = False
            # Check that New Group has Enough Peak Seperation
            if smallestNewPeak > self.xDataRMS[-1] - self.groupWidthRMS and not allPeaksCollected:
                break
            
            featureArray = []; leftBaselineArray = []
            # Take the First Feature in Each Channel's Group (Amplitude Check)
            for channelInd in range(self.numChannels):
                # Check to See if Feature was Present in the Channel
                if len(self.featureList[channelInd][currentGroupInd]) == 0:
                    channelFeature = 0; leftBase = 0
                else:
                    channelFeature = self.featureList[channelInd][currentGroupInd][0][0]
                    leftBase = self.timeDelayIndices[channelInd][currentGroupInd][0]
                    numFeatures = len(self.featureList[channelInd][currentGroupInd][0])
                # Store the Feature
                featureArray.append(channelFeature)
                leftBaselineArray.append(leftBase)

            # Check if Features are Good
            if not self.goodFeatureGroup(featureArray):
                badGroupInds += 1
                # Remove the Old Feature
                for channelInd in range(self.numChannels):
                    if self.xPeaksList[channelInd][currentGroupInd]:
                        self.badPeakInd[channelInd].append(self.xPeaksList[channelInd][currentGroupInd][0])
                    del self.xPeaksList[channelInd][currentGroupInd]
                    del self.yPeaksList[channelInd][currentGroupInd]
                    del self.featureList[channelInd][currentGroupInd]
                    del self.timeDelayIndices[channelInd][currentGroupInd]
                continue
            else:
                self.lastAnalyzedGroup += 1; 
                    
            # If the Features are Good, Move the Robot
            if True or predictionModel:
                if True and plotStreamedData:
                    maxDelay = 0; leftBase = 0
                    for channelInd in range(self.numChannels):
                        leftBaseI = self.timeDelayIndices[channelInd][currentGroupInd]
                        leftBaseI = self.xPeaksList[channelInd][currentGroupInd]
                        if leftBaseI:
                            if maxDelay < self.xDataRMS[-1] - leftBaseI[0]:
                                maxDelay = self.xDataRMS[-1] - leftBaseI[0]
                            if not leftBase or leftBaseI[0] < leftBase:
                                leftBase = leftBaseI[0]
                    for channelInd in range(self.numChannels):
                        self.timeDelayPlotsRMS[channelInd].set_data([leftBase, leftBase, self.xDataRMS[-1], self.xDataRMS[-1], leftBase], [0.01, .49, .49, 0.01, 0.01])
                        self.timeDelayPlotsRaw[channelInd].set_data([leftBase, leftBase, self.xDataRMS[-1], self.xDataRMS[-1], leftBase], [0.1, 4.9, 4.9, 0.1, 0.1])
                       # else:
                       #     self.timeDelayPlotsRMS[channelInd].set_data([],[])
                       #     self.timeDelayPlotsRaw[channelInd].set_data([],[])
                    #print("Delay Time", maxDelay)
                
                # Full Feature Array
                fullFeatureArray = []
                for numFeatureInd in range(numFeatures):
                    for channelInd in range(self.numChannels):
                        if len(self.featureList[channelInd][currentGroupInd]) == 0:
                            fullFeatureArray.append(0)
                        else:
                            fullFeatureArray.append(self.featureList[channelInd][currentGroupInd][0][numFeatureInd])
                    
                # Predict the Movement
                #self.predictMovement(fullFeatureArray, predictionModel, actionControl)
        # ------------------------------------------------------------------- #


        # --------------------------- Plot Peaks ---------------------------- #
        if plotStreamedData:
            # Plot the Peaks; Colored by Grouping
            for channelIndex in range(self.numChannels): 
                for groupNum in range(len(self.xPeaksList[channelIndex])):

                    # Check to See if the Group Has a Plot You Can Use
                    if groupNum < len(self.filteredBioelectricPeakPlots[channelIndex]):
                        groupPeakPlot = self.filteredBioelectricPeakPlots[channelIndex][groupNum]
                    # If None Availible, Create a New Plot to Add the Data
                    else:
                        channelFiltered = self.filteredBioelectricPlotAxes[channelIndex]
                        # Color Code the Group Peaks. Wrap Around to First Index When Done
                        groupColor = (groupNum-1)%(len(self.peakCurrentRightColorOrder))
                        # Create a Plot for the Peaks Using its Respective Group's Color
                        groupPeakPlot = channelFiltered.plot([], [], 'o', c=self.peakCurrentRightColorOrder[groupColor], linewidth=1, markersize = 7, alpha = 0.65)[0]
                        # Save the Plot for Later Use in the Group
                        self.filteredBioelectricPeakPlots[channelIndex].append(groupPeakPlot)
                    # Get Peak Points
                    if len(self.xPeaksList[channelIndex][groupNum]) != 0:
                        # Take the First Peak and See if You Should Plot it
                        xPeaksNew = self.xPeaksList[channelIndex][groupNum][0]
                        if self.xDataRMS[0] <= xPeaksNew:
                            # Get All the Peak
                            xPeaksNew = self.xPeaksList[channelIndex][groupNum]
                            yPeaksNew = self.yPeaksList[channelIndex][groupNum]
                            # Plot the Peaks in the Group'
                            groupPeakPlot.set_data(xPeaksNew, yPeaksNew)
                    
        # Update to Get New Data Next Round
        if plotStreamedData:
            self.fig.show()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        # -------------------------------------------------------------------#
        

    def analyzeFullBatch(self, channelNum = 1):
        print("Printing Seperate test plots")
        # Get Data to Plot
        xData = self.data['timePoints']
        yData = self.data['Channel' + str(channelNum)]
        
        # Get Data and Filter
        plt.figure()
        plt.plot(xData,yData, c='tab:blue', alpha=0.7)
        plt.title("EMG Data")
        
        plt.figure()
        filteredData = self.highPassFilter(yData)
        plt.plot(xData,filteredData, c='tab:blue', alpha=0.7)
        plt.title("Filtered Data")
        
        plt.figure()
        RMSData = self.RMSFilter(filteredData, [], self.window, self.step)
        plt.plot(xData[0:len(RMSData)],RMSData, c='tab:blue', alpha=0.7)
        plt.title("RMS Data")
        
        # Find Peaks
        batchTopPeaks = self.find_peaks(xData, RMSData)
        xPeaksNew, yPeaksNew, yBase = zip(*batchTopPeaks.items())
        xPeaksList, featureSet = self.extractFeatures(RMSData, xPeaksNew, yBase, 0)

# --------------------------------------------------------------------------- #
# ------------------------- Signal Analysis --------------------------------- #


    def highPassFilter(self, inputData):
        """
        data: Data to Filter
        f1: cutOffFreqPassThrough
        f3: cutOffFreqBand
        Rp: attDB (0.1)
        Rs: cutOffDB (30)
        samplingFreq: Frequecy You Take Data
        """            
        [n, wn] = scipy.signal.cheb1ord(self.Wp/math.pi, self.Ws/math.pi, self.Rp, self.Rs)
        [bz1, az1] = scipy.signal.cheby1(n, self.Rp, self.Wp/math.pi, 'High')
        filteredData = lfilter(bz1, az1, inputData)
        return filteredData
    
    def RMSFilter(self, inputData, RMSData = [], rmsWindow=250, stepSize=8, dataPointerRMS = 0, channelIndex = 0):
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
        # Initialize Starting Parameters
        normalization = math.sqrt(rmsWindow)
        numSteps = max(1 + math.floor((len(inputData) - rmsWindow) / stepSize), 0)
        # Take Root Mean Squared of Batch Data (numBatch = rmsWindow)
        for i in range(numSteps):
            # Get Data in the Window to take RMS
            inputWindow = inputData[i*stepSize:i*stepSize + rmsWindow]
            # Take RMS
            RMSData.append(np.linalg.norm(inputWindow, ord=2)/normalization)
            # Add to xData
            if channelIndex == 0:
                 self.xDataRMS.append(self.data['timePoints'][dataPointerRMS + i*stepSize + rmsWindow - 1])
                 
        return RMSData    


    def findPeaks(self, xData, yData, channelIndex):
        xPeaksNew = []; yPeaksNew = []; peakInds = []
    
        # For Lower Arm
        #peakIndices = scipy.signal.find_peaks(yData, prominence=.02, height=0.01, width=15, rel_height=0.4, distance = 50)[0]
        peakIndices = scipy.signal.find_peaks(yData, prominence=.007, height=0.01, width=20, rel_height=0.4, distance = 150)[0]
        # For Neck
        #peakIndices = scipy.signal.find_peaks(yData, prominence=.005, height=0.001, width=8, distance = 10)[0]
        #peakIndices = scipy.signal.find_peaks(yData, prominence=.003, height=0.0005, width=8, distance = 20)[0]
        # For Lower Leg
        #peakIndices = scipy.signal.find_peaks(yData, prominence=.003, height=0.0005, width=8, distance = 20)[0]
        # For Upper Back
        #peakIndices = scipy.signal.find_peaks(yData, prominence=.0008, height=0.0001, width=20, distance = 160)[0]
        # For Fingers (on Arm)
        #peakIndices = scipy.signal.find_peaks(yData, prominence=.002, height=0.0001, width=30, distance = 120)[0]
            
        # Find Where the New Peaks Begin
        for peakInd in peakIndices:
            xPeakLoc = xData[peakInd]
            # If it is a New Peak NOT Seen in This Channel
            if xPeakLoc not in chain(*self.xPeaksList[channelIndex]) and xPeakLoc not in self.badPeakInd[channelIndex]:
                xPeaksNew.append(xPeakLoc)
                yPeaksNew.append(yData[peakInd])
                peakInds.append(peakInd)
        
        # Return New Peaks
        return xPeaksNew, yPeaksNew, peakInds
    

    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 25, maxPointsSearch = 500):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) <= 1:
            return xPointer
        
        maxHeight = data[xPointer]; searchDirection = int(binarySearchWindow/abs(binarySearchWindow))
        # Binary Search Data to the Left to Find Minimum (Skip Over Small Bumps)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Point is Greater Than
            if data[dataPointer] > maxHeight:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, math.floor(binarySearchWindow/8), maxPointsSearch - (xPointer - dataPointer - binarySearchWindow))
            else:
                xPointer = dataPointer
                maxHeight = data[dataPointer]

        # If Your Binary Search is Too Small, Reduce it
        return self.findNearbyMinimum(data, xPointer, math.floor(binarySearchWindow/2), maxPointsSearch)
                
    def extractFeatures(self, xData, yData, peakInds):
        peakFeatures = []; leftBaselines = []
        for xPointer in peakInds:
            peakFeatures.append([])
            # Take Average of the Signal (Only Left Side As I Want to Decipher Motor Intention as Fast as I Can; Plus the Signal is generally Symmetric)
            leftBaselineIndex = max(0, self.findNearbyMinimum(yData, xPointer - 30, binarySearchWindow = -25, maxPointsSearch = 500))           # If Left Minimum Too Close to the Peak (Couldnt Find the Left Baseline)
            if leftBaselineIndex >= xPointer - 10:
                # Set the Baseline Based on the Other Peak Widths (If One Availible)
                if self.groupWidthRMSPoints:
                    leftBaselineIndex = max(0, xPointer - self.groupWidthRMSPoints)
                # Or Just Take the Last 100 Points (Good Guess)
                else:
                    leftBaselineIndex = max(0, xPointer - 100)
            # Analyze Only the Left Side (As I Want to Decipher Motor Intention as Fast as I Can; Plus the Signal is generally Symmetric)
            dataWindow = np.array(yData[leftBaselineIndex:xPointer+1])
            
            # Feature Extraction
            peakAverage = np.mean(dataWindow) - yData[leftBaselineIndex]
            peakSTD = np.std(dataWindow, ddof=1)
            peakHeight = yData[xPointer] - yData[leftBaselineIndex]
            #maxSlope = max(np.gradient(dataWindow))
            #peakEnergy = np.sum(dataWindow*dataWindow)/len(dataWindow)
            # Add Features
            peakFeatures[-1].append(peakAverage)
            peakFeatures[-1].append(peakSTD)
            peakFeatures[-1].append(peakHeight)
            #peakFeatures[-1].append(maxSlope)
            #peakFeatures[-1].append(peakEnergy)
            # Minimize Group Seperation
            if not self.groupWidthRMS:
                self.groupWidthRMS = (xData[xPointer] - xData[leftBaselineIndex])/2
                self.groupWidthRMSPoints = xPointer - leftBaselineIndex
                print("\tSetting Group Width", self.groupWidthRMS)
            
            leftBaselines.append(xData[leftBaselineIndex])
          #  print("INITIAL", xData[xPointer], leftBaselineIndex, xPointer, xData[leftBaselineIndex], self.data['timePoints'][-1])
                
        # Return Features
        return peakFeatures, leftBaselines
    
    def goodFeatureGroup(self, featureArray):
        numFeaturesFound = 0
        for feature in featureArray:
            if feature > 0:
                numFeaturesFound += 1
                
        if numFeaturesFound <= 1 and np.sum(featureArray) <= 0.1:
            print("\tOnly One Small Signal Found; Not Recording Feature");
            return False
        
        return True
    
    def predictMovement(self, inputData, predictionModel, actionControl = None): 
        # Predict Data
        predictedIndex = predictionModel.predictData(np.array([inputData]))[0]
        predictedLabel = self.gestureClasses[predictedIndex]
        print("\tThe Predicted Label is", predictedLabel)
        if actionControl:
            if predictedLabel == "left":
                actionControl.moveLeft()
            elif predictedLabel == "right":
                actionControl.moveRight()
            elif predictedLabel == "down":
                actionControl.moveDown()
            elif predictedLabel == "up":
                actionControl.moveUp()
            elif predictedLabel == "grab":
                actionControl.grabHand()
            elif predictedLabel == "release":
                actionControl.releaseHand()


"""
filteredData = readData.analysisProtocol.previousDataRMS[0]
xData = readData.analysisProtocol.data['timePoints'][-len(filteredData):]
yData = np.array(filteredData); xData = np.array(xData); filteredData = np.array(filteredData)

plt.plot(xData, filteredData)


leftIndices = []; leftIndicesOther = []
xPeaksNew, yPeaksNew, peakInds = findPeaks(xData, filteredData, 1)
for xPointer in peakInds:
    leftBaselineIndex = findNearbyMinimum(yData, xPointer, binarySearchWindow = -15, maxPointsSearch = 500)
    leftBaselineIndexOther = findBaselineIndex(xData, yData, xPointer, searchDirection = -1)
    leftIndices.append(leftBaselineIndex)
    leftIndicesOther.append(leftBaselineIndexOther)

plt.plot(xData[leftIndices], yData[leftIndices], 'o'); plt.plot(xData[leftIndicesOther], yData[leftIndicesOther], 'o')
"""

