"""
https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
"""

#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

numClassifiers = 6*(6-1)/2
    
    
    
    
    
    

class SVM:
    def __init__(self, polynomialDegree = 3):
        self.polynomialDegree = polynomialDegree
        
        # Plotting Styles
        self.stepSize = 0.01 # step size in the mesh
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red']) # Colormap
        self.cmap_bold = ['darkorange', 'c', 'darkblue', 'darkred'] # Colormap
    
    def applySVM(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):

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
            #defines how many plots: 2 rows, 2columns=> leading to 4 plots
            plt.subplot(2, 2, i + 1) #i+1 is the index
            plt.rcParams['figure.dpi'] = 300
            #space between plots
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
            #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])# Put the result into a color plot
            setPointX3 = 0.015; setPointX4 = 0.022;
            errorPoint = 0.003
            x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
            x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), x3, x4])# Put the result into a color plot
            
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)# Plot also the training points
            
            xPoints = []; yPoints = []; yLabelPoints = []
            for j,point in enumerate(Training_Data):
                if abs(point[2] - setPointX3) < errorPoint and abs(point[3] - setPointX4) < errorPoint:
                    xPoints.append(point[0])
                    yPoints.append(point[1])
                    yLabelPoints.append(Training_Labels[j])
            
            plt.scatter(xPoints, yPoints, c=yLabelPoints, cmap=plt.cm.PuBuGn, edgecolors='grey')
            
            
            #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
            
            plt.xlabel('Channel 1')
            plt.ylabel('Channel 2')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])
            
            plt.show()
        
        
        
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





