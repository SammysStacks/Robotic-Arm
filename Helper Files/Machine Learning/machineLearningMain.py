#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:47:49 2021

@author: samuelsolomon
"""
# Basic Modules
import sys
import numpy as np
import collections
# Neural Network Modules
from sklearn.model_selection import train_test_split

# Import Machine Learning Files
sys.path.append('./Machine Learning/Classification Methods/')
sys.path.append('./Classification Methods/') # Folder with Machine Learning Files
import neuralNetwork as NeuralNet       # Functions for Neural Network Algorithm
import Linear_Regression as LR          # Functions for Linear Regression Algorithm
import randomForest                     # Functions for the Random Forest Algorithm
import KNN as KNN                       # Functions for K-Nearest Neighbors' Algorithm
import SVM as SVM                       # Functions for Support Vector Machine algorithm

class predictionModelHead:
    
    def __init__(self, modelType, modelPath, dataDim, gestureClasses):
        # Store Parameters
        self.modelType = modelType
        self.modelPath = modelPath
        self.gestureClasses = gestureClasses
        self.numClasses = len(gestureClasses)
        
        # Holder Variables
        self.map2D = None
        
        # Get Prediction Model
        self.predictionModel = self.getModel(modelType, modelPath, dataDim)
        
    
    def getModel(self, modelType, modelPath, dataDim):
        # Get the Machine Learning Model
        if modelType == "NN":
            # dataDim = The dimensionality of one data point
            predictionModel = NeuralNet.Neural_Network(modelPath = modelPath, dataDim = dataDim)
        elif modelType == "RF":
            predictionModel = randomForest.randomForest(modelPath = modelPath)
        elif modelType == "LR":
            predictionModel = LR.logisticRegression(modelPath = modelPath)
        elif modelType == "KNN":
            predictionModel = KNN.KNN(modelPath = modelPath, numClasses = self.numClasses)
        elif modelType == "SVM":
            predictionModel = SVM.SVM(modelPath = modelPath, modelType = "poly", polynomialDegree = 3)
        else:
            print("No Matching Machine Learning Model was Found for '", modelType, "'");
            sys.exit()
        # Return the Precition Model
        return predictionModel
    
    def trainModel(self, signalData, signalLabels):
        # Split the Data into Training and Validation Sets
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.2, shuffle= True, stratify=signalLabels)
        signalLabelsClass = [np.argmax(i) for i in signalLabels]
        
        if self.modelType in ['NN', 'RF', 'LR', 'KNN', 'SVM']:
            # Format Labels into 1D Array (Needed for KNN Setup)
            Training_LabelsClass = [np.argmax(i) for i in Training_Labels]
            Testing_LabelsClass= [np.argmax(i) for i in Testing_Labels]
            # Train the NN with the Training Data
            self.predictionModel.trainModel(Training_Data, Training_LabelsClass, Testing_Data, Testing_LabelsClass)
            # Plot the training loss    
            self.map2D = self.predictionModel.mapTo2DPlot(signalData, signalLabelsClass)
            self.predictionModel.accuracyDistributionPlot(signalData, signalLabelsClass,  self.predictionModel.predictData(signalData), self.gestureClasses)
            self.predictionModel.plotModel(signalData, signalLabelsClass)
            self.predictionModel.plot3DLabels(signalData, signalLabelsClass)
            self.predictionModel.plot3DLabelsMovie(signalData, np.array(signalLabelsClass))

        elif self.modelType == "NN":
            # Train the NN with the Training Data
            self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, 500, seeTrainingSteps = False)
            # Plot the training loss    
            self.predictionModel.plotModel(signalData, signalLabelsClass)
            self.predictionModel.plot3DLabels(signalData, signalLabelsClass)
            self.predictionModel.accuracyDistributionPlot(signalData, signalLabelsClass,  self.predictionModel.predictData(signalData), self.gestureClasses)
            self.predictionModel.plotStats()

        # Find the Data Distribution
        classDistribution = collections.Counter(signalLabelsClass)
        print("Class Distribution:", classDistribution)
        print("Number of Data Points = ", len(classDistribution))
        
        
        