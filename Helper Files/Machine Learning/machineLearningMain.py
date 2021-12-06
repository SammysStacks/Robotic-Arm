#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:47:49 2021

@author: samuelsolomon
"""
# Basic Modules
import os
import sys
import numpy as np
import collections
import pandas as pd
from scipy import stats
# Modules for Plotting
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
# Machine Learning Modules
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
# Neural Network Modules
from sklearn.model_selection import train_test_split
# Feature Importance
import shap

# Import Machine Learning Files
sys.path.append('./Machine Learning/Classification Methods/')
sys.path.append('./Classification Methods/') # Folder with Machine Learning Files
import neuralNetwork as NeuralNet       # Functions for Neural Network Algorithm
import Linear_Regression as LR          # Functions for Linear Regression Algorithm
import randomForest                     # Functions for the Random Forest Algorithm
import KNN as KNN                       # Functions for K-Nearest Neighbors' Algorithm
import SVM as SVM                       # Functions for Support Vector Machine algorithm

sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with Machine Learning Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network

class predictionModelHead:
    
    def __init__(self, modelType, modelPath, dataDim, gestureClasses, saveDataFolder):
        # Store Parameters
        self.modelType = modelType
        self.modelPath = modelPath
        self.saveDataFolder = saveDataFolder
        self.gestureClasses = gestureClasses
        self.numClasses = len(gestureClasses)
        
        # Holder Variables
        self.map2D = []
        # Get Prediction Model
        self.predictionModel = self.getModel(modelType, modelPath, dataDim)
        if saveDataFolder:
            # Create Output File Directory to Save Data: If None
            os.makedirs(self.saveDataFolder, exist_ok=True)
        
    
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
            svmType = "poly"
            predictionModel = SVM.SVM(modelPath = modelPath, modelType = svmType, polynomialDegree = 3)
            # Section off SVM Data Analysis Into the Type of Kernels
            if self.saveDataFolder:
                self.saveDataFolder += svmType +"/"
                os.makedirs(self.saveDataFolder, exist_ok=True)
        else:
            print("No Matching Machine Learning Model was Found for '", modelType, "'");
            sys.exit()
        # Return the Precition Model
        return predictionModel
    
    def trainModel(self, signalData, signalLabels, featureLabels = []):
        if featureLabels and not len(featureLabels) == len(signalData[0]):
            print("The Number of Feature Labels Provided Does Not Match the Number of Features")
            print("Removing Feature Labels")
            featureLabels = []
            
        signalData = np.array(signalData); signalLabels = np.array(signalLabels)
        # Split the Data into Training and Validation Sets
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.33, shuffle= True, stratify=signalLabels)
        
        if self.modelType in ['RF', 'LR', 'KNN', 'SVM']:
            # Train the Model Multiple Times
            means = []
            for _ in range(3):
                modelScore = []
                # Taking the Average Score Each Time
                for _ in range(100):
                    # Train the Model with the Training Data
                    Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.33, shuffle= True, stratify=signalLabels)
                    modelScore.append(self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels))
                # Display the Spread of Scores
                plt.hist(modelScore, 100, facecolor='blue', alpha=0.5)
                # Fit the Mean Distribution and Save the Mean
                ae, loce, scalee = stats.skewnorm.fit(modelScore)
                means.append(np.round(loce*100, 2))
            # Take the Median Score as the True Score
            meanScore = np.median(means, axis=0)
            print("Mean Testing Accuracy:", meanScore)
            # Label Accuracy
            self.accuracyDistributionPlot(signalData, signalLabels,  self.predictionModel.predictData(signalData), self.gestureClasses)
            # Extract Feature Importance
            self.featureImportance(signalData, signalLabels, signalData, signalLabels, featureLabels = featureLabels, numTrials = 100)
            # Plot Model (Right Now it Only Works for 6 Gestures; Just Change the Labeling of the Classes in cmap)
            if self.numClasses == 6:
                self.plot3DLabels(signalData, signalLabels)
                #self.predictionModel.plotModel(signalData, signalLabels) # Must Edit if you Want to Use
                self.plot3DLabelsMovie(signalData, signalLabels)
                self.map2D = self.mapTo2DPlot(signalData, signalLabels)

        elif self.modelType == "NN":
            Training_LabelsArray = []; maxLabel = max(Training_Labels) + 1
            for label in Training_Labels:
                newLabel = np.zeros(maxLabel).astype(int)
                newLabel[label] = 1
                Training_LabelsArray.append(list(newLabel))
            Training_LabelsArray  = np.array(Training_LabelsArray).astype(int)

            Testing_LabelsArray = []; maxLabel = max(Training_Labels) + 1
            for label in Testing_Labels:
                newLabel = np.zeros(maxLabel).astype(int)
                newLabel[label] = 1
                Testing_LabelsArray.append(list(newLabel))
            Testing_LabelsArray  = np.array(Testing_LabelsArray).astype(int)
                
            # Train the NN with the Training Data
            self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Training_Labels, 500, seeTrainingSteps = False)
            # Plot the training loss    
            self.predictionModel.plotModel(signalData, signalLabels)
            self.predictionModel.plot3DLabels(signalData, signalLabels)
            self.predictionModel.accuracyDistributionPlot(signalData, signalLabels,  self.predictionModel.predictData(signalData), self.gestureClasses)
            self.predictionModel.plotStats()

        # Find the Data Distribution
        classDistribution = collections.Counter(signalLabels)
        print("Class Distribution:", classDistribution)
        print("Number of Data Points = ", len(classDistribution))
        
        
    def mapTo2DPlot(self, signalData, signalLabels, name = "Channel Map"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(signalData, signalLabels)
        
        mds = MDS(n_components=2,random_state=0, n_init = 4)
        X_2d = mds.fit_transform(X_scaled)
        
        X_2d = self.rotatePoints(X_2d, -np.pi/2).T
        
        figMap = plt.scatter(X_2d[:,0], X_2d[:,1], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 130, marker='.', edgecolors='k')        
        
        # Figure Aesthetics
        fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
        figMap.set_clim(-0.5, 5.5)
        plt.title('Channel Feature Map');
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
        
        return X_2d
    
    def rotatePoints(self, rotatingMatrix, theta_rad = -np.pi/2):

        A = np.matrix([[np.cos(theta_rad), -np.sin(theta_rad)],
                       [np.sin(theta_rad), np.cos(theta_rad)]])
        
        m2 = np.zeros(rotatingMatrix.shape)
        
        for i,v in enumerate(rotatingMatrix):
          w = A @ v.T
          m2[i] = w
        m2 = m2.T
        
        return m2
    
    
    def plot3DLabels(self, signalData, signalLabels, name = "Channel Feature Distribution"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        ax = plt.axes(projection='3d')
        
        # Scatter Plot
        ax.scatter3D(signalData[:, 3], signalData[:, 1], signalData[:, 2], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 100, edgecolors='k')
        
        ax.set_title('Channel Feature Distribution');
        ax.set_xlabel("Channel 4")
        ax.set_ylabel("Channel 2")
        ax.set_zlabel("Channel 3")
        #fig.tight_layout()
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
    
    def plot3DLabelsMovie(self, signalData, signalLabels, name = "Channel Feature Distribution Movie"):
        # Plot and Save
        fig = plt.figure()
        #fig.set_size_inches(15,15,10)
        ax = plt.axes(projection='3d')
        
        # Initialize Relevant Channel 4 Range
        errorPoint = 0.01; # Width of Channel 4's Values
        channel4Vals = np.arange(min(signalData[:, 3]), max(signalData[:, 3]), 2*errorPoint)
        
        # Initialize Movie Writer for Plots
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=name + " " + self.modelType, artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=2, metadata=metadata)
        
        with writer.saving(fig, self.saveDataFolder + name + " " + self.modelType + ".mp4", 300):
            for channel4Val in channel4Vals:
                channelPoints1 = signalData[:, 0][abs(signalData[:, 3] - channel4Val) < errorPoint]
                channelPoints2 = signalData[:, 1][abs(signalData[:, 3] - channel4Val) < errorPoint]
                channelPoints3 = signalData[:, 2][abs(signalData[:, 3] - channel4Val) < errorPoint]
                currentLabels = signalLabels[abs(signalData[:, 3] - channel4Val) < errorPoint]
                
                if len(currentLabels) != 0:
                    # Scatter Plot
                    figMap = ax.scatter3D(channelPoints1, channelPoints2, channelPoints3, "o", c = currentLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 50, edgecolors='k')
        
                    ax.set_title('Channel Feature Distribution; Channel 4 = ' + str(channel4Val) + " ± " + str(errorPoint));
                    ax.set_xlabel("Channel 1")
                    ax.set_ylabel("Channel 2")
                    ax.set_zlabel("Channel 3")
                    ax.yaxis._axinfo['label']['space_factor'] = 20
                    
                    ax.set_xlim3d(0, max(signalData[:, 0]))
                    ax.set_ylim3d(0, max(signalData[:, 1]))
                    ax.set_zlim3d(0, max(signalData[:, 2]))
                    
                    # Figure Aesthetics
                    cb = fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
                    plt.rcParams['figure.dpi'] = 300
                    figMap.set_clim(-0.5, 5.5)
                    
                    # Write to Video
                    writer.grab_frame()
                    # Clear Previous Frame
                    plt.cla()
                    cb.remove()
                
        plt.show() # Must be the Last Line
    
    def accuracyDistributionPlot(self, signalData, signalLabelsTrue, signalLabelsML, movementOptions, name = "Accuracy Distribution"):
        
        # Calculate the Accuracy Matrix
        accMat = np.zeros((len(movementOptions), len(movementOptions)))
        for ind, channelFeatures in enumerate(signalData):
            # Sum(Row) = # of Gestures Made with that Label
            # Each Column in a Row = The Number of Times that Gesture Was Predicted as Column Label #
            accMat[signalLabelsTrue[ind]][signalLabelsML[ind]] += 1
        
        # Scale Each Row to 100
        for label in range(len(movementOptions)):
            accMat[label] = 100*accMat[label]/np.sum(accMat[label])
        
        # Make plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8,8)
        
        # Make heatmap on plot
        im, cbar = createMap.heatmap(accMat, movementOptions, movementOptions, ax=ax,
                           cmap="copper", cbarlabel="Gesture Accuracy (%)")
        createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        font = {'family' : 'verdana',
                'weight' : 'bold',
                'size'   : 8}
        matplotlib.rc('font', **font)
        
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=130, bbox_inches='tight')
        plt.show()
    
    
    def plotImportance(self, perm_importance_result, featureLabels, name = "Relative Feature Importance"):
        """ bar plot the feature importance """
    
        fig, ax = plt.subplots()
    
        indices = perm_importance_result['importances_mean'].argsort()
        plt.barh(range(len(indices)),
                 perm_importance_result['importances_mean'][indices],
                 xerr=perm_importance_result['importances_std'][indices])
    
        ax.set_yticks(range(len(indices)))
        if featureLabels:
            _ = ax.set_yticklabels(np.array(featureLabels)[indices])
      #      headers = np.array(featureLabels)[indices]
      #      for i in headers:
      #          print('%s Weight: %.5g' % (str(i),v))
        plt.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=150, bbox_inches='tight')
        
    
    def featureImportance(self, signalData, signalLabels, Testing_Data, Testing_Labels, featureLabels = [], numTrials = 100):
        """
        Randomly Permute a Feature's Column and Return the Average Deviation in the Score: |oldScore - newScore|
        NOTE: ONLY Compare Feature on the Same Scale: Time and Distance CANNOT be Compared
        """
        importanceResults = permutation_importance(self.predictionModel.model, signalData, signalLabels, n_repeats=numTrials)
        self.plotImportance(importanceResults, featureLabels)
        
        if self.modelType == "RF":
            # get importance
            importance = self.predictionModel.model.feature_importances_
            # summarize feature importance
            for i,v in enumerate(importance):
                if featureLabels:
                    i = featureLabels[i]
                    print('%s Weight: %.5g' % (str(i),v))
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            freq_series = pd.Series(importance)
            ax = freq_series.plot(kind="bar")
            
            # Specify Figure Aesthetics
            ax.set_title("Feature Importance in Model")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Feature Importance")
            
            # Set X-Labels
            if featureLabels:
                ax.set_xticklabels(featureLabels)
                self.add_value_labels(ax)
            # Show Plot
            name = "Feature Importance"
            plt.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=150, bbox_inches='tight')
            pyplot.show()
            
            explainer = shap.TreeExplainer(self.predictionModel.model)
            shap_values = explainer.shap_values(signalData)
            
            shap.summary_plot(shap_values, signalData, plot_type="bar")
            
            shap.summary_plot(shap_values, signalData)
        
        
        if featureLabels:
            # Make Output Folder for SHAP Values
            os.makedirs(self.saveDataFolder + "SHAP Values/", exist_ok=True)
            # Create Panda DataFrame to Match Input Type for SHAP
            testingDataPD = pd.DataFrame(Testing_Data, columns = featureLabels)
            
            # More General Explainer
            explainerGeneral = shap.Explainer(self.predictionModel.model.predict, testingDataPD)
            shap_valuesGeneral = explainerGeneral(testingDataPD)
            
            # MultiClass (Only For Tree)
            if self.modelType == "RF":
                explainer = shap.TreeExplainer(self.predictionModel.model)
                shap_values = explainer.shap_values(testingDataPD)
            else:
                # Calculate Shap Values
                explainer = shap.KernelExplainer(self.predictionModel.model.predict, testingDataPD)
                shap_values = explainer.shap_values(testingDataPD, nsamples=len(signalData))
            
            # Specify Indivisual Sharp Parameters
            dataPoint = 10
            featurePoint = 20
            
            # Summary Plot
            name = "Summary Plot"
            summaryPlot = plt.figure()
            if self.modelType == "RF":
                shap.summary_plot(shap_valuesGeneral, testingDataPD, plot_type="bar", class_names=self.gestureClasses, feature_names = featureLabels)
            else:
                shap.summary_plot(shap_valuesGeneral, testingDataPD, class_names=self.gestureClasses, feature_names = featureLabels)
            summaryPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Dependance Plot
            name = "Dependance Plot"
            dependancePlot, dependanceAX = plt.subplots()
            shap.dependence_plot(featurePoint, shap_values, features = testingDataPD, feature_names = featureLabels, ax = dependanceAX)
            dependancePlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Indivisual Force Plot
            name = "Indivisual Force Plot"
            forcePlot = shap.force_plot(explainer.expected_value, shap_values[dataPoint,:], features = np.round(testingDataPD.iloc[dataPoint,:], 5), feature_names = featureLabels, matplotlib = True, show = False)
            forcePlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Full Force Plot. NOTE: CANNOT USE matplotlib = True to See
            name = "Full Force Plot"
            fullForcePlot = shap.force_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = featureLabels, matplotlib = False, show = True)
            shap.save_html(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".htm", fullForcePlot)
            
            # WaterFall Plot
            name = "Waterfall Plot"
            waterfallPlot = plt.figure()
            #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0dataPoint], feature_names = featureLabels, max_display = len(featureLabels), show = True)
            shap.plots.waterfall(shap_valuesGeneral[dataPoint],  max_display = len(featureLabels), show = True)
            waterfallPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
 
            # Indivisual Decision Plot
            misclassified = Testing_Labels != self.predictionModel.model.predict(Testing_Data)
            decisionFolder = self.saveDataFolder + "SHAP Values/Decision Plots/"
            os.makedirs(decisionFolder, exist_ok=True) 
            for dataPoint1 in range(min(50, len(testingDataPD))):
                name = "Indivisual Decision Plot DataPoint Num " + str(dataPoint1)
                decisionPlot = plt.figure()
                shap.decision_plot(explainer.expected_value, shap_values[dataPoint1,:], features = testingDataPD.iloc[dataPoint1,:], feature_names = featureLabels, feature_order = "importance", highlight = misclassified[dataPoint1])
                decisionPlot.savefig(decisionFolder + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Decision Plot
            name = "Decision Plot"
            decisionPlotOne = plt.figure()
            shap.decision_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = featureLabels, feature_order = "importance", highlight = misclassified)
            decisionPlotOne.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Bar Plot
            name = "Bar Plot"
            barPlot = plt.figure()
            shap.plots.bar(shap_valuesGeneral, max_display = len(featureLabels), show = True)
            barPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)

            # HeatMap Plot
            name = "Heatmap Plot"
            heatmapPlot = plt.figure()
            shap.plots.heatmap(shap_valuesGeneral, max_display = len(featureLabels), show = True, instance_order=shap_valuesGeneral.sum(1))
            heatmapPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
                
            # Scatter Plot
            scatterFolder = self.saveDataFolder + "SHAP Values/Scatter Plots/"
            os.makedirs(scatterFolder, exist_ok=True)
            for featurePoint1 in range(len(featureLabels)):
                for featurePoint2 in range(len(featureLabels)):
                    name = "Scatter Plot (" + featureLabels[featurePoint1] + " VS " + featureLabels[featurePoint2] + ")" 
                    scatterPlot, scatterAX = plt.subplots()
                    shap.plots.scatter(shap_valuesGeneral[:, featureLabels[featurePoint1]], color = shap_valuesGeneral[:, featureLabels[featurePoint2]], ax = scatterAX)
                    scatterPlot.savefig(scatterFolder + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Monitoring Plot (The Function is a Beta Test As of 11-2021)
            if len(Testing_Data) > 150:  # They Skip Every 50 Points I Believe
                name = "Monitor Plot"
                monitorPlot = plt.figure()
                shap.monitoring_plot(featurePoint, shap_values, features = testingDataPD, feature_names = featureLabels)
                monitorPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
                        
    def add_value_labels(self, ax, spacing=5):
        """Add labels to the end of each bar in a bar chart.
    
        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """
    
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
    
            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'
    
            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'
    
            # Use Y value as label and format number with one decimal place
            label = "{:.3f}".format(y_value)
    
            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.
        
        