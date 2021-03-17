"""
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors


class KNN:
    
    def __init__(self, numClasses):
        self.numNeighbors = numClasses
        
        # Plotting Styles
        self.stepSize = 0.01 # step size in the mesh
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red']) # Colormap
        self.cmap_bold = ['darkorange', 'c', 'darkblue', 'darkred'] # Colormap
    
    def applyKNN(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, weightDistances = ['uniform', 'distance']):  
                
        for weight in weightDistances:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors=self.numNeighbors, weights=weight, algorithm='auto', 
                        leaf_size=30, p=1, metric='minkowski', metric_params=None, n_jobs=None)
            clf.fit(Training_Data, Training_Labels)
            print("Score with weight =", weight, ":", clf.score(Testing_Data, Testing_Labels))
        
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = Training_Data[:, 0].min(), Training_Data[:, 0].max()
            y_min, y_max = Training_Data[:, 1].min(), Training_Data[:, 1].max()
            xx, yy = np.meshgrid(np.arange(x_min, x_max, self.stepSize),
                                 np.arange(y_min, y_max, self.stepSize))
            
            #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            setPointX3 = 0.15; setPointX4 = 0.12;
            errorPoint = 0.03
            x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
            x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), x3, x4])# Put the result into a color plot
                    
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            
            plt.rcParams['figure.dpi'] = 300
            plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)# Plot also the training points
            
            xPoints = []; yPoints = []; yLabelPoints = []
            for j, point in enumerate(Training_Data):
                if abs(point[2] - setPointX3) < errorPoint and abs(point[3] - setPointX4) < errorPoint:
                    xPoints.append(point[0])
                    yPoints.append(point[1])
                    yLabelPoints.append(Training_Labels[j])
            
            #plt.scatter(xPoints, yPoints, alpha=1.0, edgecolor="black", linewidths=2)
            plt.scatter(xPoints, yPoints, c=yLabelPoints, cmap=plt.cm.PuBuGn, edgecolors='grey', s=50)
        
            # Plot also the training points
            #sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y],
            #                palette=cmap_bold, alpha=1.0, edgecolor="black")
            
            
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("Classification (k = %i, weights = '%s')"
                      % (self.numNeighbors, weight))
            plt.xlabel('Channel 1')
            plt.ylabel('Channel 2')
        
            plt.show()
            
            
            