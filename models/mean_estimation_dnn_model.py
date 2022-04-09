import pandas as pd
import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Flatten, Activation, Dense)
import matplotlib.pyplot as plt

from utils.preprocessing import high_resolution_coordinates

class Callback(tf.keras.callbacks.Callback):
    epoch_controller = 25

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if (self.epoch%self.epoch_controller==0):
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
        
def dnn_model(X_train, 
          Y_train, 
          num_layers=3,
          neurons_per_layer=1000,
          weight_initializer='standard_normal',
          activation_per_layer='relu',
          epochs=500, 
          loss='mse', 
          l1_regularizer=True,
          l1_penalty=1e-7,
          batch_size=4,
          verbose=0):
    """
    create and return the desired model
    """
    #regularizer
    if l1_regularizer:
        regularizer=tf.keras.regularizers.L1(l1=l1_penalty)
    else:
        regularizer=None
        
    #kernel initializers to initialize weights
    if weight_initializer=='standard_normal':
        initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1)
    if weight_initializer=='he_normal':
        initializer=tf.keras.initializers.HeNormal()
        
    # add max norm constraint
    max_norm = tf.keras.constraints.MaxNorm(max_value=2, axis=0)
        
    #model definition
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    for n in range(num_layers):
        model.add(Dense(neurons_per_layer, 
                        kernel_initializer=initializer, 
                        kernel_regularizer=regularizer, 
                        kernel_constraint=max_norm,
                        activation=activation_per_layer))
    model.add(Dense(1))
    
            
    #define loss to minimize
    if loss=='huber':
        loss_to_minimize = tf.keras.losses.Huber() 
    elif loss=='mse':
        loss_to_minimize = tf.keras.losses.MeanSquaredError()
    
    #optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, decay=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, 
        amsgrad=False, name='Adam'
    )
    
    #compile the model
    model.compile(loss=loss_to_minimize, 
                  optimizer=optimizer)
    
    #model fit
    model_history = model.fit(X_train, Y_train, 
                              batch_size=batch_size, 
                              epochs=epochs,  
                              callbacks=[Callback()], 
                              verbose=verbose)
    
    return model_history, model


def train_predict(X, Y, **model_params):
    """
    Train and predict pixel values from the model
    """
    num_layers = model_params["num_layers"] if "num_layers" in model_params else 3
    neurons_per_layer = model_params["neurons_per_layer"] if "neurons_per_layer" in model_params else 1000
    weight_initializer = model_params["weight_initializer"] if "weight_initializer" in model_params else "standard_normal"
    activation_per_layer = model_params["activation_per_layer"] if "activation_per_layer" in model_params else "relu"
    epochs = model_params["epochs"] if "epochs" in model_params else 500
    loss = model_params["loss"] if "loss" in model_params else "mse"
    l1_regularizer = model_params["l1_regularizer"] if "l1_regularizer" in model_params else True
    l1_penalty = model_params["l1_penalty"] if "l1_penalty" in model_params else 1e-7
    batch_size = model_params["batch_size"] if "batch_size" in model_params else 4
    verbose = model_params["verbose"] if "verbose" in model_params else 0
    get_high_resolution_image = model_params["get_high_resolution_image"] if "get_high_resolution_image" in model_params else False
    high_resolution_dimensions = model_params["high_resolution_dimensions"] if "high_resolution_dimensions" in model_params else (95,79)
    cmap = model_params["cmap"] if "cmap" in model_params else "gray"
    save_image_location_and_name = model_params["save_image_location_and_name"] if "save_image_location_and_name" else False
    
    model_history, model = dnn_model(X, Y,
                                     num_layers=num_layers,
                                     neurons_per_layer=neurons_per_layer,
                                     weight_initializer=weight_initializer,
                                     activation_per_layer=activation_per_layer, 
                                     epochs=epochs, 
                                     loss=loss, 
                                     l1_regularizer=l1_regularizer, 
                                     l1_penalty=l1_penalty,
                                     verbose=verbose)
    
    if get_high_resolution_image:
        num_rows, num_cols = high_resolution_dimensions
        X = high_resolution_coordinates(num_rows=num_rows, num_cols=num_cols)
        y_pred = model.predict(X)
        if save_image_location_and_name:
            plt.imshow(np.reshape(y_pred, (num_rows, num_rows)), cmap=cmap) #display the recovered image
            plt.savefig(save_image_location_and_name)
    else:
        y_pred = model.predict(X)
        default_dims = (50, 50)
        if save_image_location_and_name:
            plt.imshow(np.reshape(y_pred, (default_dims[0], default_dims[1])), cmap=cmap) #display the recovered image
            plt.savefig(save_image_location_and_name)
        
    return model_history, model, y_pred
