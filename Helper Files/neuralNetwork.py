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
# Import Packages]
import tensorflow as tf
import numpy as np
#from tensorflow import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
# Plotting
import matplotlib.pyplot as plt

from tensorflow.python.keras.utils import losses_utils
# ---------------------------------------------------------------------------#
# ------------------------- Neural Network ----------------------------------#

class Helpers:
    def __init__(self, name, dataDimension, numClasses = 5, optimizer=None, lossFuncs=None, metrics=None):
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


class Neural_Network:
    """
    Define a Neural Network Class
    """
    def __init__(self, name, dim, opt=None, loss=None, metric=None):
        """
        Input:
            name: The Name of the Neural Network to Save/Load
            dim: The dimension of 1 data point (# of columns in data)
        Output: None
        Save: model, name
        """
        # Tries to find a compiled model identical to name (in same folder)
        if os.path.exists(name):
            with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                model = load_model(name)
            print("Previous Neural Network Loaded")
        # If unsuccessful, create a new model
        else:
            # Define a TensorFlow Neural Network using Keras
                # Sequential: Inputing the List of Hidden Layers into the Network (in order)
                # Dense: Adds a layer of neurons
                    # (unit = # neurons in layer, activation function, *if first layer* shape of input data)
                # input_shape: The dimension of 1 Data Point (# of rows in one column)
            model = tf.keras.Sequential()
            # Model Layers
            
            #model.add(tf.keras.layers.Reshape((1,4)))
            #model.add(tf.keras.layers.LSTM(256))
            
            #model.add(tf.keras.layers.Dropout(.02, input_shape=(dim,)))
            
            model.add(tf.keras.layers.Dense(units=4, activation=tf.nn.tanh))
            #model.add(tf.keras.layers.Dense(units=3, activation=tf.nn.tanh))
            
            #model.add(tf.keras.layers.Dropout(.02, input_shape=(dim,)))
            
            model.add(tf.keras.layers.Dense(units=6, activation='softmax'))
            
            
            # Define the Loss Function and Optimizer for the Model
                # Compile: Initializing the optimizer and the loss in the Neural Network
                # Optimizer: The method used to change the Weights in the Network
                # Loss: The Function used to estimate how bad our weights are
            if opt == None:
                opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            if loss == None:
                loss = 'mean_absolute_error'
            if metric == None:
                metric = ['categorical_hinge', 'mae', 'categorical_crossentropy']
            self.opt = opt
            self.loss = loss
            self.metric = metric
            #sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
            model.compile(optimizer = opt, loss = loss, metrics = list([metric]))
            print("New Neural Network Created")
        self.model = model
        self.name = name


    
    def train_model(self, Training_Data, Training_Labels, epochs = 500, seeTrainingSteps = True):
        # For mini-batch gradient decent we want it small (not full batch) to better generalize data
        max_batch_size = 32                    # Keep Batch sizes relatively small (no more than 32 or 64)
        mini_batch_gd = min(len(Training_Data)//4, max_batch_size)
        mini_batch_gd = max(1, mini_batch_gd)  # For really small data samples at least take 1 data point
        # For every Epoch (loop), run the Neural Network by:
            # With uninitialized weights, bring data through network
            # Calculate the loss based on the data
            # Perform optimizer to update the weights
        history = self.model.fit(Training_Data, Training_Labels, validation_split=0.33,
                                 epochs=epochs, shuffle=True, batch_size = mini_batch_gd, verbose = seeTrainingSteps)
        return history
    
    
    
    def neural_net_prediction(self, New_Data, New_Labels = 0, printResults = False):
        # Predict Label based on new Data
        if printResults:
            print("New Data Predictions \n", self.model.predict(New_Data))
        return self.model.predict(New_Data)
        
    
    
    def save_model(self, outputNueralNetwork):
        self.model.save(outputNueralNetwork)  # creates a HDF5 file 'my_model.h5'    
    
    
    
    def plot_statistics(self, history):
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        # plot accuracy during training
        #pyplot.subplot(212)
        #pyplot.title('Accuracy')
        #pyplot.plot(history.history['accuracy'], label='train')
        #pyplot.plot(history.history['val_accuracy'], label='test')
        #pyplot.legend()
        pyplot.show()



if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    
    # Saving the Neural network
    SaveNeuralNetwork = True
    saveNeuralNetworkName = "testNet2"
    saveNeuralNetworkFolder = "./Neural Network/"
    
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Machine learning Program (Should Not Have to Edit)           #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    # Input Training Data and Labels
    # Training_Data: Every row is one data point, Dim(row) = dimensionality of data
    data = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],
                     [14,14],[15,15],[16,16],[17,17],[18,18],[29,19],[20,20]])
    labels = np.array([[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]]).T
    # Split into Training and Validation Data
    Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(
                            data, labels, test_size=0.2, shuffle= True)
    # Find the Dimension of the data
    rows, cols = Training_Data.shape
    
    # Display the Training Data
    plt.figure()
    plt.plot(Training_Data[:,1], Training_Labels.flatten(), 'r')
    plt.show()
    
    # Make the Neural Network
    outputNeuralNetwork = saveNeuralNetworkFolder+saveNeuralNetworkName
    nn = Neural_Network(name = outputNeuralNetwork, dim = cols) # dim = The dimensionality of one data point    
    # Train the network
    Neural_Network_Statistics = nn.train_model(Training_Data, Training_Labels, 100)
    # Make a prediction using new data
    nn.neural_net_prediction(Testing_Data, Testing_Labels)
    # Plot the training loss    
    nn.plot_statistics(Neural_Network_Statistics)
    # Save the Neural Network for Later Use
    if SaveNeuralNetwork:
        nn.save_model()
    
    # Display the Testing Data
    plt.figure()
    plt.plot(Testing_Data[:,1], Testing_Labels.flatten(), 'b')
    plt.show()




