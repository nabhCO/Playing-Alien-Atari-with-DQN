import keras
import tensorflow as tf
import numpy as np

#took a lot of inspo from the Keras "Model" class documentation (for creating a subclass specifically)
class ConvNeuralNet(keras.Model):

    def __init__(self):

        super().__init__()

        self.input_layer = keras.Input(shape=(4, 84, 84, 3), batch_size=32)
        self.conv_layer_1 = keras.layers.Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu")
        self.conv_layer_2 = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu")
        self.dense_hidden_layer = keras.layers.Dense(units=256, activation="relu")
        self.output_layer = keras.layers.Dense(units = 18, activation = "linear")

    def call(self, input):

        input_layer_output = self.input_layer(input)
        conv_layer_1_output = self.conv_layer_1(input_layer_output)
        conv_layer_2_output = self.conv_layer_2(conv_layer_1_output)
        dense_hidden_layer_output = self.dense_hidden_layer(conv_layer_2_output)
        return self.output_layer(dense_hidden_layer_output)



class DQNAgent:

    def __init__(self, gamma, alpha, epsilon):

        #add epsilon decay later

        optimizer = keras.optimizers.RMSprop(learning_rate=alpha)
        loss_fn = keras.losses.MeanSquaredError()

    




        

