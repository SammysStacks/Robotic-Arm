"""
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py


"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import matplotlib.animation as manimation
import joblib
from sklearn.model_selection import train_test_split

sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network

class logisticRegression:
    
    def __init__(self, modelPath):
        
        # Plotting Styles
        self.stepSize = 0.01 # step size in the mesh
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red']) # Colormap
        self.cmap_bold = ['darkorange', 'c', 'darkblue', 'darkred'] # Colormap
        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel()
    
    def saveModel(self, modelPath = "./LR.sav"):
        joblib.dump(self.model, 'scoreregression.pkl')    
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("LR Model Loaded")
            
    
    def createModel(self):
        self.model = LogisticRegression()
        print("LR Model Created")
        
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):  
        # Train the Model
        self.model.fit(Training_Data, Training_Labels)
        self.scoreModel(Testing_Data, Testing_Labels)
    
    def scoreModel(self, signalData, signalLabels):
        print("Score:", self.model.score(signalData, signalLabels))
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
    
    def plot3DLabels(self, signalData, signalLabels, saveFolder = "../Output Data/", name = "Channel Feature Distribution"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(10,10)
        ax = plt.axes(projection='3d')
        
        # Scatter Plot
        ax.scatter(signalData[:, 3], signalData[:, 1], signalData[:, 2], "o", c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', 6), linewidth = 0.2, s = 30)
        
        ax.set_title('Channel Feature Distribution');
        ax.set_xlabel("Channel 4")
        ax.set_ylabel("Channel 2")
        ax.set_zlabel("Channel 3")
        #fig.tight_layout()
        fig.savefig(saveFolder + name + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
        
    def accuracyDistributionPlot(self, signalData, signalLabelsTrue, signalLabelsML, movementOptions, saveFolder = "../Output Data/", name = "Accuracy Distribution"):
        
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
                'size'   : 9}
        matplotlib.rc('font', **font)
        
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(saveFolder + name + ".png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plotModel(self, signalData, signalLabels):
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.2, shuffle= True, stratify=signalLabels)
                
        # we create an instance of Linear Regression Model
        model = LogisticRegression()
        model.fit(Training_Data, Training_Labels)
        print("Score:", model.score(Testing_Data, Testing_Labels))
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = Training_Data[:, 0].min(), Training_Data[:, 0].max()
        y_min, y_max = Training_Data[:, 1].min(), Training_Data[:, 1].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.stepSize),
                             np.arange(y_min, y_max, self.stepSize))
        
        
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title="", artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=3, metadata=metadata)
        
        
        
        setPointX4 = 0.002;
        errorPoint = 0.003;
        dataWithinChannel4 = Training_Data[abs(Training_Data[:,3] - setPointX4) <= errorPoint]
        
        channel3Vals = np.arange(0.0, dataWithinChannel4[:,2].max(), 0.01)
        fig = plt.figure()
        
        with writer.saving(fig, "LogisticRegression.mp4", 300):
            for setPointX3 in channel3Vals:
            
                #Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                # setPointX3 = 0.15; setPointX4 = 0.12;
                x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
                x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
                Z = model.predict(np.c_[xx.ravel(), yy.ravel(), x3, x4])# Put the result into a color plot
                        
                # Put the result into a color plot                    
                Z = Z.reshape(xx.shape)

                plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('cubehelix', 6), alpha=0.7, vmin=0, vmax=5)
                
                xPoints = []; yPoints = []; yLabelPoints = []
                for j, point in enumerate(Training_Data):
                    if abs(point[2] - setPointX3) <= errorPoint and abs(point[3] - setPointX4) <= errorPoint:
                        xPoints.append(point[0])
                        yPoints.append(point[1])
                        yLabelPoints.append(Training_Labels[j])
                
                plt.scatter(xPoints, yPoints, c=yLabelPoints, cmap=plt.cm.get_cmap('cubehelix', 6), edgecolors='grey', s=50, vmin=0, vmax=5)
                
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                #plt.title("Classification (k = %i, weights = '%s')"
                #          % (self.numNeighbors, weight))
                plt.title("Channel3 = " + str(round(setPointX3,3)) + "; Channel4 = " + str(setPointX4) + "; Error = " + str(errorPoint))
                plt.xlabel('Channel 1')
                plt.ylabel('Channel 2')
                plt.rcParams['figure.dpi'] = 300
                
                cb = plt.colorbar(ticks=range(6), label='digit value')
                plt.clim(-0.5, 5.5)
            
                # Write to Video
                writer.grab_frame()
                plt.cla()
                cb.remove()
        
        # Save Model
                
            
            