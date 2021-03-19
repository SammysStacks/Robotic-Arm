"""
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import matplotlib.animation as manimation
import joblib

class logisticRegression:
    
    def __init__(self):
        
        # Plotting Styles
        self.stepSize = 0.01 # step size in the mesh
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red']) # Colormap
        self.cmap_bold = ['darkorange', 'c', 'darkblue', 'darkred'] # Colormap
    
    def saveModel(self, model, modelPath = "./KNN.sav"):
        joblib.dump(model, 'scoreregression.pkl')
    
    def loadModel(self, modelPath):
        model = joblib.load(modelPath, mmap_mode ='r')
        return model
    
    def applyLR(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, saveModel = False):  
                
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
                
            
            