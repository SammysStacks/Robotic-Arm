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
            self.scoreModel(newData, newLabels, scoreType)
    
    def scoreModel(self, signalData, signalLabels, scoreType = "Score:"):
        print(scoreType, self.model.score(signalData, signalLabels))
        
        # Evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
        n_scores = cross_val_score(self.model, signalData, signalLabels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # report performance
        print('Cross-Validation Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
        
    def accuracyDistributionPlot(self, signalData, trueLabels, predicatedLabels, signalLabelsTitles, saveFolder = "./Output Data/Machine Learning Results/", name = "Accuracy Distribution"):
        
        # Calculate the Accuracy Matrix
        accMat = np.zeros((len(signalLabelsTitles), len(signalLabelsTitles)))
        for ind, channelFeatures in enumerate(signalData):
            # Sum(Row) = # of Gestures Made with that Label
            # Each Column in a Row = The Number of Times that Gesture Was Predicted as Column Label #
            accMat[trueLabels[ind]][predicatedLabels[ind]] += 1
        
        # Scale Each Row to 100
        for label in range(len(signalLabelsTitles)):
            accMat[label] = 100*accMat[label]/np.sum(accMat[label])
        
        # Make plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8,8)
        
        # Make heatmap on plot
        im, cbar = createMap.heatmap(accMat, signalLabelsTitles, signalLabelsTitles, ax=ax,
                           cmap="copper", cbarlabel="Label Accuracy (%)")
        createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        font = {'family' : 'verdana',
                'weight' : 'bold',
                'size'   : 9}
        matplotlib.rc('font', **font)
                
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(saveFolder + name + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        
        
    def mapTo2DPlot(self, signalData, signalLabels, saveFolder = "../Output Data/", name = "Channel Map"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        
        # Scatter Plot
        #plt.scatter(signalData[:, 0]-signalData[:, 1] + signalData[:, 2]-signalData[:, 3], signalData[:, 0]-signalData[:, 2] + signalData[:, 1]-signalData[:, 3], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', 6), s = 130, marker='.', edgecolors='k')        
        #plt.scatter(signalData[:, 0]**2-signalData[:, 1]**2 + signalData[:, 2]**2-signalData[:, 3]**2, signalData[:, 0]**2-signalData[:, 2]**2 + signalData[:, 1]**2-signalData[:, 3]**2, c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', 6), s = 130, marker='.', edgecolors='k')
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(signalData, signalLabels)
        
        mds = MDS(n_components=2,random_state=0, n_init = 4)
        X_2d = mds.fit_transform(X_scaled)
        
        X_2d = self.rotatePoints(X_2d, -np.pi/2).T
        
        figMap = plt.scatter(X_2d[:,0], X_2d[:,1], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', 6), s = 130, marker='.', edgecolors='k')        
        
        # Figure Aesthetics
        fig.colorbar(figMap, ticks=range(6), label='digit value')
        figMap.set_clim(-0.5, 5.5)
        plt.title('Channel Feature Map');
        #plt.xlabel("Channel 1+2")
        #plt.ylabel("Channel 3+4")
        #fig.tight_layout()
        fig.savefig(saveFolder + name + ".png", dpi=200, bbox_inches='tight')
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

    def plot3DLabels(self, signalData, signalLabels, saveFolder = "../Output Data/", name = "Channel Feature Distribution"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        ax = plt.axes(projection='3d')
        
        # Scatter Plot
        ax.scatter3D(signalData[:, 3], signalData[:, 1], signalData[:, 2], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', 6), s = 100, edgecolors='k')
        
        ax.set_title('Channel Feature Distribution');
        ax.set_xlabel("Channel 4")
        ax.set_ylabel("Channel 2")
        ax.set_zlabel("Channel 3")
        #fig.tight_layout()
        fig.savefig(saveFolder + name + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
    
    def plot3DLabelsMovie(self, signalData, signalLabels, saveFolder = "../Output Data/", name = "Channel Feature Distribution Movie"):
        # Plot and Save
        fig = plt.figure()
        #fig.set_size_inches(15,15,10)
        ax = plt.axes(projection='3d')
        
        # Initialize Relevant Channel 4 Range
        errorPoint = 0.01; # Width of Channel 4's Values
        channel4Vals = np.arange(min(signalData[:, 3]), max(signalData[:, 3]), 2*errorPoint)
        
        # Initialize Movie Writer for Plots
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=name, artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=2, metadata=metadata)
        
        with writer.saving(fig, saveFolder + name + ".mp4", 300):
            for channel4Val in channel4Vals:
                channelPoints1 = signalData[:, 0][abs(signalData[:, 3] - channel4Val) < errorPoint]
                channelPoints2 = signalData[:, 1][abs(signalData[:, 3] - channel4Val) < errorPoint]
                channelPoints3 = signalData[:, 2][abs(signalData[:, 3] - channel4Val) < errorPoint]
                currentLabels = signalLabels[abs(signalData[:, 3] - channel4Val) < errorPoint]
                
                if len(currentLabels) != 0:
                    # Scatter Plot
                    figMap = ax.scatter3D(channelPoints1, channelPoints2, channelPoints3, "o", c = currentLabels, cmap = plt.cm.get_cmap('cubehelix', 6), s = 50, edgecolors='k')
        
                    ax.set_title('Channel Feature Distribution; Channel 4 = ' + str(channel4Val) + " ± " + str(errorPoint));
                    ax.set_xlabel("Channel 1")
                    ax.set_ylabel("Channel 2")
                    ax.set_zlabel("Channel 3")
                    ax.yaxis._axinfo['label']['space_factor'] = 20
                    
                    ax.set_xlim3d(0, max(signalData[:, 0]))
                    ax.set_ylim3d(0, max(signalData[:, 1]))
                    ax.set_zlim3d(0, max(signalData[:, 2]))
                    
                    # Figure Aesthetics
                    cb = fig.colorbar(figMap, ticks=range(6), label='digit value')
                    plt.rcParams['figure.dpi'] = 300
                    figMap.set_clim(-0.5, 5.5)
                    
                    # Write to Video
                    writer.grab_frame()
                    # Clear Previous Frame
                    plt.cla()
                    cb.remove()
                
        plt.show() # Must be the Last Line
        

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


