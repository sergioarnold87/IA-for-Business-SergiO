#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 08:57:13 2021

@author: sergio
"""

# Importar las librerías
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# Construccion del cerebro
class Brain(object):
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(units = 64, activation = "sigmoid")(states)
        y = Dense(units = 32, activation = "sigmoid")(x)
        q_values = Dense(units = number_actions, activation = "softmax")(y)
        self.model = Model(inputs = states, output = q_values)
        self.model.compile(loss = "mse", optimizer = Adam(lr = learning_rate))
        
     