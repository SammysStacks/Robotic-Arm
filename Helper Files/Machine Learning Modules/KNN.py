"""
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import matplotlib.animation as manimation
import joblib
import os

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
        print("Score:", self.model.score(Testing_Data, Testing_Labels))
    
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
        setPointX4 = 0.002; # Channel 4's Value
        errorPoint = 0.003; # Width of Channel 4's Values
        x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
        dataWithinChannel4 = signalData[abs(signalData[:, 3] - setPointX4) <= errorPoint]
        # Initialize Relevant Channel 3 Range
        channel3Vals = np.arange(0.0, dataWithinChannel4[:,2].max(), 0.01)
        
        # Plot Data with Different Channel 3 Values, and Save as Movie
        with writer.saving(fig, "./Machine Learning Modules/ML Videos/KNN_" + self.weight + ".mp4", 300):
            for setPointX3 in channel3Vals:
                # Define New Channel 3 Points
                x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
                
                # Predict Every Point's Class With the Trained Model
                handMovements = self.model.predict(np.c_[xx.ravel(), yy.ravel(), x3, x4])
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
            
            