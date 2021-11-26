"""
Code Written by Samuel Solomon

SKLearn SVM Guide: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Import Basic Modules
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Import Modules for Plotting
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# Import Machine Learning Modules
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold

# Import Python Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network
    
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class randomForest:
    def __init__(self, modelPath):        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel()
    
    def saveModel(self, modelPath = "./SVM.sav"):
        joblib.dump(self.model, modelPath)    
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("Random Forest Model Loaded")
            
    def createModel(self):
        """
        criteria: “gini” for the Gini impurity and “entropy” for the information gain.
        """
        self.model = RandomForestClassifier(n_estimators=100,criterion='gini', max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',max_leaf_nodes=None,
                    min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                    verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        print("Random Forest Model Created")
        
    def trainModel(self, Training_Data, Training_Labels, newData = [], newLabels = [], scoreType = "Score:"):
        # Train the Model
        self.model.fit(Training_Data, Training_Labels)
        # Score the Model
        if len(newData) > 0:
            modelScore = self.scoreModel(newData, newLabels, scoreType)
            return modelScore
    
    def scoreModel(self, signalData, signalLabels, scoreType = "Score:"):
        return self.model.score(signalData, signalLabels)
        
        # Evaluate the model
    #    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    #    n_scores = cross_val_score(self.model, signalData, signalLabels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    #    # report performance
    #    print('Cross-Validation Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
        

    def plotModel(self, signalData, signalLabels):
    
        # Create Mesh (X,Y Points) to Predict Classifier's Space
        stepSize = 0.01 # step size in the mesh
        x_min, x_max = signalData[:, 0].min(), signalData[:, 0].max()  # Channel 1
        y_min, y_max = signalData[:, 1].min(), signalData[:, 1].max()  # Channel 2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, stepSize), np.arange(y_min, y_max, stepSize))
        
        # Define MovieWriter to Mave Movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title="", artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=3, metadata=metadata)
        # Define Movie Plot
        fig = plt.figure()
        
        # Set Channel4 as Constant: Display Data ONLY in This Range
        setPointX4 = 0.1; # Channel 4's Value
        errorPoint = 0.005; # Width of Channel 4's Values
        x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
        dataWithinChannel4 = signalData[abs(signalData[:, 3] - setPointX4) <= errorPoint]
        # Initialize Relevant Channel 3 Range
        channel3Vals = np.arange(0.0, dataWithinChannel4[:,2].max(initial=0), 0.01)
        if len(channel3Vals) == 0:
            print("No Values Found in Channel 4")
            return None
        
        # Plot Data with Different Channel 3 Values, and Save as Movie
        with writer.saving(fig, "./Machine Learning Modules/ML Videos/KNN_" + self.weight + ".mp4", 300):
            for setPointX3 in channel3Vals:
                # Define New Channel 3 Points
                x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
                
                # Predict Every Point's Class With the Trained Model
                handMovements = self.predictData(np.c_[xx.ravel(), yy.ravel(), x3, x4])
                # Rearrange the Data to Match the 2D Plot                  
                handMovements = handMovements.reshape(xx.shape)
                # Plot the Predicted Classes
                plt.contourf(xx, yy, handMovements, cmap=plt.cm.get_cmap('cubehelix', 6), alpha=0.7, vmin=0, vmax=5)
                
                # Get the Real Data with the Real Labels in this Space
                xPoints = []; yPoints = []; yLabelPoints = []
                for j, point in enumerate(signalData):
                    if abs(point[2] - setPointX3) <= errorPoint and abs(point[3] - setPointX4) <= errorPoint:
                        xPoints.append(point[0])
                        yPoints.append(point[1])
                        yLabelPoints.append(signalLabels[j])
                # Plot the Real Data with the Real Labels in this Space
                plt.scatter(xPoints, yPoints, c=yLabelPoints, cmap=plt.cm.get_cmap('cubehelix', 6), edgecolors='grey', s=50, vmin=0, vmax=5)
                
                # Figure Labels/Limits
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.xlabel('Channel 1')
                plt.ylabel('Channel 2')
                plt.title("Channel3 = " + str(round(setPointX3,3)) + "; Channel4 = " + str(setPointX4) + "; Error = " + str(errorPoint))
                
                # Figure Aesthetics
                cb = plt.colorbar(ticks=range(6), label='digit value')
                plt.rcParams['figure.dpi'] = 300
                plt.clim(-0.5, 5.5)
                
                # Write to Video
                writer.grab_frame()
                # Clear Previous Frame
                plt.cla()
                cb.remove()


    def plotImportance(self, perm_importance_result, headerTitles):
        """ bar plot the feature importance """
    
        fig, ax = plt.subplots()
    
        indices = perm_importance_result['importances_mean'].argsort()
        plt.barh(range(len(indices)),
                 perm_importance_result['importances_mean'][indices],
                 xerr=perm_importance_result['importances_std'][indices])
    
        ax.set_yticks(range(len(indices)))
        if headerTitles:
            _ = ax.set_yticklabels(np.array(headerTitles)[indices])
    
    def featureImportance(self, signalData, signalLabels, headerTitles = [], numTrials = 30):
        """
        Randomly Permute a Feature's Column and Return the Average Deviation in the Score: |oldScore - newScore|
        NOTE: ONLY Compare Feature on the Same Scale: Time and Distance CANNOT be Compared
        """
        importanceResults = permutation_importance(self.model, signalData, signalLabels, n_repeats=numTrials)
        self.plotImportance(importanceResults, headerTitles)
        
        
        # get importance
        importance = self.model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            if headerTitles:
                i = headerTitles[i]
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
        if headerTitles:
            ax.set_xticklabels(headerTitles)
            self.add_value_labels(ax)
        # Show Plot
        pyplot.show()
        
        
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


