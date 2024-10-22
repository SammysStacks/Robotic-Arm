r"""
@author: Sam

Installation:
    $ conda install tensorflow
    $ conda install keras
    $ conda install numpy
    $ conda install matplotlib
    

Citation:
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}}
  }
"""

# ------------------------ Imported Packages --------------------------------#


import os
import sys
import numpy as np
import matplotlib
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import matplotlib.pyplot as plt
# Import Packages]
import tensorflow as tf
#from tensorflow import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from matplotlib import pyplot

from tensorflow.python.keras.utils import losses_utils
import itertools

sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network

# ---------------------------------------------------------------------------#
# ------------------------- Neural Network ----------------------------------#

class Helpers:
    def __init__(self, name, dataDimension, numClasses = 6, optimizer=None, lossFuncs=None, metrics=None):
        self.name = name
        self.dataDimension = dataDimension
        self.numClasses = numClasses
        if optimizer:
            self.optimizers = list(optimizer)
        else:
            self.optimizers = [
                tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'),
                tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad'),
                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam'),
                tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'),
               # tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
               # tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop'),
               # tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
               # tf.keras.optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5,
               #        initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0,
               #         name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
                ]
        if lossFuncs:
            self.loss = list(lossFuncs)
        else:
            self.loss = [
                tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO, name='binary_crossentropy'),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO, name='categorical_crossentropy'),
                tf.keras.losses.CategoricalHinge(reduction=losses_utils.ReductionV2.AUTO, name='categorical_hinge'),
                tf.keras.losses.CosineSimilarity(axis=-1, reduction=losses_utils.ReductionV2.AUTO, name='cosine_similarity'),
                tf.keras.losses.Hinge(reduction=losses_utils.ReductionV2.AUTO, name='hinge'),
                tf.keras.losses.Huber(delta=1.0, reduction=losses_utils.ReductionV2.AUTO, name='huber_loss'),
                tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.AUTO, name='kl_divergence'),
                tf.keras.losses.LogCosh(reduction=losses_utils.ReductionV2.AUTO, name='log_cosh'),
                tf.keras.losses.Loss(reduction=losses_utils.ReductionV2.AUTO, name=None),
                tf.keras.losses.MeanAbsoluteError(reduction=losses_utils.ReductionV2.AUTO, name='mean_absolute_error'),
                tf.keras.losses.MeanAbsolutePercentageError(reduction=losses_utils.ReductionV2.AUTO, name='mean_absolute_percentage_error'),
                tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_error'),
                tf.keras.losses.MeanSquaredLogarithmicError(reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_logarithmic_error'),
                tf.keras.losses.Poisson(reduction=losses_utils.ReductionV2.AUTO, name='poisson'),
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.AUTO, name='sparse_categorical_crossentropy'),
                tf.keras.losses.SquaredHinge(reduction=losses_utils.ReductionV2.AUTO, name='squared_hinge'),
                ]
        if metrics:
            self.metrics = list(metrics)
        else:
            self.metrics = [
                tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None),
                tf.keras.metrics.Accuracy(name='accuracy', dtype=None),
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5),
                tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy', dtype=None, from_logits=False, label_smoothing=0),
                tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None),
                tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy', dtype=None, from_logits=False, label_smoothing=0),
                tf.keras.metrics.CategoricalHinge(name='categorical_hinge', dtype=None),
                tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1),
                tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None),
                tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None),
                tf.keras.metrics.Hinge(name='hinge', dtype=None),
                tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None),
                tf.keras.metrics.LogCoshError(name='logcosh', dtype=None),
                tf.keras.metrics.Mean(name='mean', dtype=None),
                tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None),
                tf.keras.metrics.MeanAbsolutePercentageError(name='mean_absolute_percentage_error', dtype=None),
                tf.keras.metrics.MeanIoU(num_classes=numClasses, name=None, dtype=None),
                tf.keras.metrics.MeanRelativeError(normalizer=[1]*dataDimension, name=None, dtype=None),
                tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None),
                tf.keras.metrics.MeanSquaredLogarithmicError(name='mean_squared_logarithmic_error', dtype=None),
                tf.keras.metrics.MeanTensor(name='mean_tensor', dtype=None),
                tf.keras.metrics.Poisson(name='poisson', dtype=None),
                tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                tf.keras.metrics.PrecisionAtRecall(recall=0.5, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                tf.keras.metrics.RecallAtPrecision(precision=0.8, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error', dtype=None),
                tf.keras.metrics.SensitivityAtSpecificity(specificity=0.5, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy', dtype=None, from_logits=False, axis=-1),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='sparse_top_k_categorical_accuracy', dtype=None),
                tf.keras.metrics.SpecificityAtSensitivity(sensitivity=0.5, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.SquaredHinge(name='squared_hinge', dtype=None),
                tf.keras.metrics.Sum(name='sum', dtype=None),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_k_categorical_accuracy', dtype=None),
                tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None),
                tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None),
                ]
        
    def neuralPermutations(self):
        neuralOptimizerList = []
        for opt in self.optimizers:
            for loss in self.loss:
                for metric in self.metrics:
                    neuralOptimizerList.append(Neural_Network(self.name, self.dataDimension, opt, loss, metric))
        return neuralOptimizerList
    
    def permuteMetrics(self, opt, loss):
        neuralOptimizerList = []
        for metric in itertools.permutations(self.metrics, 2):
            neuralOptimizerList.append(Neural_Network(self.name, self.dataDimension, opt, loss, list(metric)))
        return neuralOptimizerList


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
    
class Neural_Network:
    """
    Define a Neural Network Class
    """
    def __init__(self, modelPath, dataDim):
        """
        Input:
            name: The Name of the Neural Network to Save/Load
        Output: None
        Save: model, name
        """
        # Define Model Parameters
        self.history = None
        
        # Initialize Model
        self.model = None
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel(dataDim, opt=None, loss=None, metric=None)


    def loadModel(self, modelPath):
        # Tries to find a compiled model identical to name (in same folder)
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.model = load_model(modelPath)
        print("NN Model Loaded")
    
    def createModel(self, dataDim, opt=None, loss=None, metric=None):
        """
        Parameters
        ----------
        dataDim : The dimension of 1 data point (# of columns in data)
        opt : Neural Network Optimizer
        loss : Neural Network Loss Function
        metric : Neurala Network Metric to Score Accuracy
        """
        # Define a TensorFlow Neural Network using Keras
            # Sequential: Input the List of Hidden Layers into the Network (in order)
            # Dense: Adds a layer of neurons
                # (unit = # neurons in layer, activation function, *if first layer* shape of input data)
            # Input_shape: The dimension of 1 Data Point (# of rows in one column)
        self.model = tf.keras.Sequential()
        
        # Model Layers
        #model.add(tf.keras.layers.Reshape((1,4)))
        #model.add(tf.keras.layers.LSTM(256))
        self.model.add(tf.keras.layers.Dense(units=dataDim, activation='sigmoid'))
        #model.add(tf.keras.layers.Dense(units=8, activation=tf.nn.tanh))
        #model.add(tf.keras.layers.Dense(units=12, activation=tf.nn.tanh))
        #model.add(tf.keras.layers.Dense(units=8, activation=tf.nn.tanh))
        #model.add(tf.keras.layers.Dense(units=3, activation=tf.nn.tanh))
        #model.add(tf.keras.layers.Dropout(.02, input_shape=(dim,)))
        self.model.add(tf.keras.layers.Dense(units=6, activation='softmax'))
        
        # Define the Loss Function and Optimizer for the Model
            # Compile: Initializing the optimizer and the loss in the Neural Network
            # Optimizer: The method used to change the Weights in the Network
            # Loss: The Function used to estimate how bad our weights are
        if opt == None:
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if loss == None:
            loss = 'cosine_similarity'
        if metric == None:
            metric = ['accuracy', 'logcosh']
        
        # Compile the Model
        self.model.compile(optimizer = opt, loss = loss, metrics = list([metric]))
        print("NN Model Created")
    
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, epochs = 500, seeTrainingSteps = False):
        # For mini-batch gradient decent we want it small (not full batch) to better generalize data
        max_batch_size = 128  # Keep Batch sizes relatively small (no more than 64 or 128)
        mini_batch_gd = min(len(Training_Data)//4, max_batch_size)
        mini_batch_gd = max(1, mini_batch_gd)  # For really small data samples at least take 1 data point
        # For every Epoch (loop), run the Neural Network by:
            # With uninitialized weights, bring data through network
            # Calculate the loss based on the data
            # Perform optimizer to update the weights
        print(Training_Data, Training_Labels)
        self.history = self.model.fit(Training_Data, Training_Labels, validation_split=0.33, epochs=int(epochs), shuffle=True, batch_size = int(mini_batch_gd), verbose = seeTrainingSteps)
        # Score the Model
        results = self.model.evaluate(Testing_Data, Testing_Labels, batch_size=mini_batch_gd, verbose = seeTrainingSteps)
        score = results[0]; accuracy = results[1]; 
        print('Test score:', score)
        print('Test accuracy:', accuracy)
    
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        predictionProbs = self.model.predict(New_Data)
        return np.argmax(predictionProbs, axis=1)
    
    def saveModel(self, outputNueralNetwork):
        self.model.save(outputNueralNetwork)  # creates a HDF5 file 'my_model.h5'    
    
    
    def plotStats(self):
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        # plot accuracy during training
        #pyplot.subplot(212)
        #pyplot.title('Accuracy')
        #pyplot.plot(history.history['accuracy'], label='train')
        #pyplot.plot(history.history['val_accuracy'], label='test')
        #pyplot.legend()
        pyplot.show()
        
        
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
        channel3Vals = np.arange(0.0, dataWithinChannel4[:,2].max(initial=0), 0.01)
        if len(channel3Vals) == 0:
            print("No Values Found in Channel 3")
            return None
        
        # Plot Data with Different Channel 3 Values, and Save as Movie
        with writer.saving(fig, "./Machine Learning Modules/ML Videos/NN.mp4", 300):
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
                        yLabelPoints.append(np.argmax(signalLabels[j]))
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



