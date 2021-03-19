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
#from tensorflow import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from matplotlib import pyplot

from tensorflow.python.keras.utils import losses_utils
import itertools

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
        model = tf.keras.Sequential()
        
        # Model Layers
        #model.add(tf.keras.layers.Reshape((1,4)))
        #model.add(tf.keras.layers.LSTM(256))
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
            loss = 'categorical_hinge'
        if metric == None:
            metric = ['logcosh', 'mae', 'categorical_crossentropy']
        
        # Compile the Model
        model.compile(optimizer = opt, loss = loss, metrics = list([metric]))
        # Store the Model in the Class
        self.model = model
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
        self.history = self.model.fit(Training_Data, Training_Labels, validation_split=0.33,
                                 epochs=epochs, shuffle=True, batch_size = mini_batch_gd, verbose = seeTrainingSteps)
        # Score the Model
        results = self.model.evaluate(Testing_Data, Testing_Labels, batch_size=mini_batch_gd, verbose = seeTrainingSteps)
        score = results[0]; accuracy = results[1:]
        print('Test score:', score)
        print('Test accuracy:', accuracy)
    
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
    
    def saveModel(self, outputNueralNetwork):
        self.model.save(outputNueralNetwork)  # creates a HDF5 file 'my_model.h5'    
    
    
    def plotModel(self):
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



