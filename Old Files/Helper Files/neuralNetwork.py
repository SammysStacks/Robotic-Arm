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


# Import Packages
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


# ---------------------------------------------------------------------------#
# ------------------------- Neural Network ----------------------------------#


class Neural_Network:
    """
    Define a Neural Network Class
    """
    def __init__(self, name, dim):
        """
        Input:
            name: The Name of the Neural Network to Save/Load
            dim: The dimension of 1 data point (# of columns in data)
        Output: None
        Save: model, name
        """
        # Tries to find a compiled model identical to name (in same folder)
        try:
            with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                model = load_model(name)
            print("Previous Neural Network Loaded")
        # If unsuccessful, create a new model
        except:
            # Define a TensorFlow Neural Network using Keras
                # Sequential: Inputing the List of Hidden Layers into the Network (in order)
                # Dense: Adds a layer of neurons
                    # (unit = # neurons in layer, activation function, *if first layer* shape of input data)
                # input_shape: The dimension of 1 Data Point (# of rows in one column)
            model = tf.keras.Sequential()
            # Model Layers
            
            model.add(tf.keras.layers.Dense(units=4, input_shape=[dim]), activation=tf.nn.sigmoid)
            model.add(tf.keras.layers.LSTM(128))
            model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
            #model.add(tf.keras.layers.Dropout(.05, input_shape=(dim,)))
            
            # Define the Loss Function and Optimizer for the Model
                # Compile: Initializing the optimizer and the loss in the Neural Network
                # Optimizer: The method used to change the Weights in the Network
                # Loss: The Function used to estimate how bad our weights are
            model.compile(optimizer = 'adam',
                          loss = 'categorical_crossentropy', metrics=['accuracy'])
            print("New Neural Network Created")
        self.model = model
        self.name = name


    
    def train_model(self, Training_Data, Training_Labels, epochs = 500):
        # For mini-batch gradient decent we want it small (not full batch) to better generalize data
        max_batch_size = 32                    # Keep Batch sizes relatively small (no more than 32 or 64)
        mini_batch_gd = min(len(Training_Data)//3, max_batch_size)
        mini_batch_gd = max(1, mini_batch_gd)  # For really small data samples at least take 1 data point
        print("Batch Size: ", mini_batch_gd)
        # For every Epoch (loop), run the Neural Network by:
            # With uninitialized weights, bring data through network
            # Calculate the loss based on the data
            # Perform optimizer (SGD) to update the weights
        history = self.model.fit(Training_Data, Training_Labels, validation_split=0.2,
                                 epochs=epochs, shuffle=True, batch_size = mini_batch_gd)
        return history
    
    
    
    def neural_net_prediction(self, New_Data, New_Labels = 0, printResults = False):
        # Predict Label based on new Data
        if printResults:
            print("New Data Predictions \n", self.model.predict(New_Data))
            print("New Data Deviation from Predicted Labels \n", self.model.predict(New_Data) - New_Labels)
        return self.model.predict(New_Data)
        
    
    
    def save_model(self):
        self.model.save(self.name)  # creates a HDF5 file 'my_model.h5'    
    
    
    
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




