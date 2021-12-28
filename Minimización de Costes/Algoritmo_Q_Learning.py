#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 11:45:44 2021

@author: sergio
"""

# Importar las librerías
import numpy as np

# IMPLEMENTAR EL DEEP Q-LEARNING CON REPETICIÓN DE EXPERIENCIA

class DQN(object):
    
    # INTRODUCIR E INICIALIZAR TODOS LOS PARÁMETROS Y VARIABLES DEL DQN
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount
    
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA LA MEMORIA DE LA REPETICIÓN DE EXPERIENCIA
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # CREACIÓN DEL MÉTODO QUE COSTRUYE DOS LOTES DE ENTRADAS Y OBJETIVOS
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        targets = np.zeros((min(len_memory, batch_size), num_outputs))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size = min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            input[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i,action] = reward + self.discount * Q_sa
        return inputs, targets
    
    
