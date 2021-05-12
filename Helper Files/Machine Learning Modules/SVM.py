"""
https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
"""

#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.animation as manimation
import joblib
import sys
import os
from sklearn.model_selection import train_test_split


sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network
    

class SVM:
    def __init__(self, modelPath, modelType = "rbf", polynomialDegree = 3):
        self.polynomialDegree = polynomialDegree
        
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
            self.createModel(modelType)
    
    def saveModel(self, modelPath = "./SVM.sav"):
        joblib.dump(self.model, 'scoreregression.pkl')    
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("SVM Model Loaded")
            
    
    def createModel(self, modelType = "rbf"):
        if modelType == "linear":
            self.model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
        elif modelType == "rbf":
            self.model = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')
        elif modelType == "poly":
            self.model = svm.SVC(kernel='poly', degree = self.polynomialDegree, C=1, decision_function_shape='ovo')
        elif modelType == "sigmoid":
            self.model = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')
        else:
            print("No SVM Model Matches the Requested Type")
            sys.exit()
        print("SVM Model Created")
        
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

        linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        poly = svm.SVC(kernel='poly', degree = self.polynomialDegree, C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
    
        #to better understand it, just play with the value, change it and print it
        x_min, x_max = Training_Data[:, 0].min(), Training_Data[:, 0].max()
        y_min, y_max = Training_Data[:, 1].min(), Training_Data[:, 1].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.stepSize),np.arange(y_min, y_max, self.stepSize))# create the title that will be shown on the plot
        titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
        
        """
        dimensions = []
        for dimension in range(np.shape(X_train)[1]):
            x_min, x_max = X[:, dimension].min() - 1, X[:, dimension].max() + 1
            dimensions.append(np.arange(x_min, x_max, h))
        xx, yy = np.meshgrid(*dimensions) # create the title that will be shown on the plot
        titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
        """
        
        for i, clf in enumerate((linear, rbf, poly, sig)):
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title="", artist='Matplotlib', comment='Movie support!')
            writer = FFMpegWriter(fps=3, metadata=metadata)
            
            setPointX4 = 0.002;
            errorPoint = 0.003;
            dataWithinChannel4 = Training_Data[abs(Training_Data[:,3] - setPointX4) <= errorPoint]
            
            channel3Vals = np.arange(0.0, dataWithinChannel4[:,2].max(), 0.01)
            
            #defines how many plots: 2 rows, 2columns=> leading to 4 plots
            fig = plt.figure()
            plt.rcParams['figure.dpi'] = 300
            #space between plots
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
            with writer.saving(fig, "./Machine Learning Modules/ML Videos/SVM_" + clf.kernel + ".mp4", 300):
                for setPointX3 in channel3Vals:
            
                    x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
                    x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
                    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), x3, x4])# Put the result into a color plot
                    
                    Z = Z.reshape(xx.shape)
                    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('cubehelix', 6), alpha=0.7, vmin=0, vmax=5)
                    
                    xPoints = []; yPoints = []; yLabelPoints = []
                    for j, point in enumerate(Training_Data):
                        if abs(point[2] - setPointX3) <= errorPoint and abs(point[3] - setPointX4) <= errorPoint:
                            xPoints.append(point[0])
                            yPoints.append(point[1])
                            yLabelPoints.append(Training_Labels[j])
                    
                    plt.scatter(xPoints, yPoints, c=yLabelPoints, cmap=plt.cm.get_cmap('cubehelix', 6), edgecolors='grey', s=50, vmin=0, vmax=5)
                    
                    
                    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
                    plt.title(titles[i]+": Channel3 = " + str(round(setPointX3,3)) + "; Channel4 = " + str(setPointX4) + "; Error = " + str(errorPoint))
                    plt.xlabel('Channel 1')
                    plt.ylabel('Channel 2')
                    plt.xlim(xx.min(), xx.max())
                    plt.ylim(yy.min(), yy.max())
                    plt.xticks(())
                    plt.yticks(())
                    #plt.title(titles[i])
                    
                    cb = plt.colorbar(ticks=range(6), label='digit value')
                    plt.clim(-0.5, 5.5)
                
                    # Write to Video
                    writer.grab_frame()
                    plt.cla()
                    cb.remove()
        
        
        
        # retrieve the accuracy and print it for all 4 kernel functions
        accuracy_lin = linear.score(Testing_Data, Testing_Labels)
        accuracy_poly = poly.score(Testing_Data, Testing_Labels)
        accuracy_rbf = rbf.score(Testing_Data, Testing_Labels)
        accuracy_sig = sig.score(Testing_Data, Testing_Labels)
        
        print("Accuracy Linear Kernel:", accuracy_lin)
        print("Accuracy Polynomial Kernel:", accuracy_poly)
        print("Accuracy Radial Basis Kernel:", accuracy_rbf)
        print("Accuracy Sigmoid Kernel:", accuracy_sig) 
        
        linear_pred = linear.predict(Testing_Data)
        poly_pred = poly.predict(Testing_Data)
        rbf_pred = rbf.predict(Testing_Data)
        sig_pred = sig.predict(Testing_Data) 
        
        # creating a confusion matrix
        cm_lin = confusion_matrix(Testing_Labels, linear_pred)
        cm_poly = confusion_matrix(Testing_Labels, poly_pred)
        cm_rbf = confusion_matrix(Testing_Labels, rbf_pred)
        cm_sig = confusion_matrix(Testing_Labels, sig_pred)
        
        print(cm_lin)
        print(cm_poly)
        print(cm_rbf)
        print(cm_sig)





