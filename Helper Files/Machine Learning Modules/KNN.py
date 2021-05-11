"""
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import matplotlib
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import joblib
import os

sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network

class KNN:
    def __init__(self, modelPath, numClasses, weight = 'distance'):
        self.numNeighbors = numClasses
        self.weight = weight
        self.model = None
        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel(weight)
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("KNN Model Loaded")
    
    def createModel(self, weight):
        self.model = neighbors.KNeighborsClassifier(n_neighbors = self.numNeighbors, weights = weight, algorithm = 'auto', 
                        leaf_size = 30, p = 1, metric = 'minkowski', metric_params = None, n_jobs = None)
        print("KNN Model Created")
        
    def saveModel(self, modelPath = "./KNN.pkl"):
        with open(modelPath, 'wb') as handle:
            joblib.dump(self.model, handle)
    
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):  
        # Train the Model
        self.model.fit(Training_Data, Training_Labels)
        self.scoreModel(Testing_Data, Testing_Labels)
    
    def scoreModel(self, signalData, signalLabels):
        print("Score:", self.model.score(signalData, signalLabels))
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
    
    def plot3DLabels(self, signalData, signalLabels):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,15)
        ax = plt.axes(projection='3d')
        
        # Scatter Plot
        ax.scatter(signalData[:, 3], signalData[:, 1], signalData[:, 2], "o", c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', 6), linewidth = 0.1, s = 20)
        # Connected Plot
        #ax.plot_trisurf(time, potential, current, cmap='viridis', edgecolor='none')
        
        ax.set_title('Channel Feature Distribution');
        ax.set_xlabel("Channel 4")
        ax.set_ylabel("Channel 2")
        ax.set_zlabel("Channel 3")
        fig.tight_layout(pad=5)
        #fig.savefig(outputData + base + ".png", dpi=300)
        plt.show() # Must be the Last Line
        
    def accuracyDistributionPlot(self, signalData, signalLabelsTrue, signalLabelsML, movementOptions):
        signalLabelsTrue = [np.argmax(i) for i in signalLabelsTrue]
        
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
        fig.set_size_inches(15,15)
        
        # Make heatmap on plot
        print("\n" + "Making Heat Map")
        im, cbar = createMap.heatmap(accMat, movementOptions, movementOptions, ax=ax,
                           cmap="copper", cbarlabel="Gesture Accuracy (%)")
        print("Adding Annotations")
        texts = createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        print("Styling Fonts")
        font = {'family' : 'verdana',
                'weight' : 'bold',
                'size'   : 20}
        matplotlib.rc('font', **font)
        
        # Format, save, and show
        print("Tightening Layout")
        #fig.tight_layout()
        print("saving")
        plt.savefig("heat_map.png", dpi=300, bbox_inches='tight')
        print("showing")
        plt.show()

    
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
            
            