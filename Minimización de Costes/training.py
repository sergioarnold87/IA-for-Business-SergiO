#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 12:43:34 2021

@author: sergio
"""

# Entrenamiento de la IA

# Instalación de Keras
# conda install -c conda-forge keras

# Importar las librerías y el resto de ficheros de python
import os
import numpy as np
import random as rn
import Entorno
import cerebro
import Algoritmo_Q_Learning

# Establecer semillas para la reproducibilidad del experimento
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURACIÓN DE LOS PARÁMETROS
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT CLASS
env = Entorno.Environment(optimal_temperature = (18.0, 24.0),
                              initial_month = 0,
                              initial_number_users = 20,
                              initial_rate_data = 30)

# CONSTRUCCIÓN DEL CEREBRO CREADO UN OBJETO DE LA CLASE BRAIN
brain = cerebro.Brain(learning_rate = 0.00001, number_actions = number_actions)

# CONSTRUCCIÓN DEL MODELO DE DQN CREANDO UN OBJETO DE LA CLASE DQN 
dqn = Algoritmo_Q_Learning.DQN(max_memory = max_memory, discount = 0.9)

# ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = True

# Entrenamiento de la IA
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

if (env.train):
    #Arrancar el bulce sobre todas las epochs ( 1 Epoch = 5 Meses)
    for epoch in range(1, number_epochs):
        #Inicializar las variables del Environment como del bucle de entrenamiento
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        
        # INICIALIZACIÓN DEL BUCLE DE TIMESTEPS (Timestep = 1 minuto) EN UNA EPOCA
        while ((not game_over) and (timestep <= 5*30*24*60)):
            # EJECUTAR LA SIGUIENTE ACCIÓN POR EXPLORACIÓN
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                   
            # EJECUTAR LA SIGUIENTE ACCIÓN POR INFERENCIA
            else: 
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
            
            if (action < direction_boundary):
                direction = -1
            else:
                direction = 1
            energy_ai = abs(action - direction_boundary) * temperature_step
        # ACTUALIZAR EL ENTORNO Y ALCANZAR EL SIGUIENTE ESTADO
        next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
        total_reward += reward   
        
        # ALMACENAR LA NUEVA TRANSICIÓN EN LA MEMORIA
        dqn.remember([current_state, action, reward, next_state], game_over)
        
        #OBTENER LOS DOS BLOQUES SEPARADOS DE ENTRADAS Y OBJETIVOS
        
        inputs, targets = dqn.get_batch(model, batch_size = batch_size)
        
        # CALCULAR LA PÉRDIDA EN LOS DOS LOTES DE ENTRADAS Y OBJETIVOS
        loss += model.train_on_batch(inputs, targets)
        timestep += 1
        current_state = next_state   
        # IMPRIMIR EL RESULTADO DE ENTREAMIENTO PARA CADA EPOCH

    print("\n")
    print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
    print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
    print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
    
    # DETENCIÓN TEMPRANA
    if early_stopping:
        if (total_reward <= best_total_reward):
            patience_count += 1
        else:
            best_total_reward = total_reward
            patience_count = 0
            
        if patience_count >= patience:
            print("Ejecución prematura del método")
            break
        
    # GUARDAR EL MODELO PARA SU USO FUTURO
    model.save("model.h5")


