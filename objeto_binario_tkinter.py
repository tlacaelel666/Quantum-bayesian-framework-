"""Este script crea una aplicación de escritorio con una interfaz gráfica (GUI) que simula un entorno simple y permite entrenar a un "agente" inteligente para interactuar con él usando técnicas de Aprendizaje por Refuerzo (Reinforcement Learning - RL).

fecha 30-03-25
autor: Jacobo Tlacaelel Mina Rodríguez 
proyecto cuadrante-QuoreMind v1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import re
import numpy as np
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque  # Import deque for memory replay
from torch.distributions import Categorical # Needed for A2C

# Configuración de logging mejorada
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logger's level

# Eliminar handlers existentes si se recarga el script (útil en algunos entornos)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a file handler and set level to debug
log_file_path = 'quantum_agent.log'
if os.path.exists(log_file_path):
    try:
        os.remove(log_file_path)
        print(f"Archivo de log anterior eliminado: {log_file_path}")
    except PermissionError as e:
        print(f"No se pudo eliminar el archivo de log existente {log_file_path}: {e}. Continuando sin eliminar.")
    except Exception as e:
        print(f"Error inesperado al intentar eliminar {log_file_path}: {e}")

fh = logging.FileHandler(log_file_path, encoding='utf-8')
fh.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Show INFO and above in console

# Add formatter to handlers
fh.setFormatter(CustomFormatter())
ch.setFormatter(CustomFormatter())

# Add handlers to the logger
if not logger.handlers: # Evitar duplicar handlers si el script se ejecuta múltiples veces
    logger.addHandler(fh)
    logger.addHandler(ch)

# Handler para el área de texto en Tkinter
class LogTextHandler(logging.Handler):
    def __init__(self, text_area):
        logging.Handler.__init__(self)
        self.text_area = text_area

    def emit(self, record):
        msg = self.format(record)
        def append_message():
            self.text_area.configure(state='normal')
            self.text_area.insert(tk.END, msg + '\n')
            self.text_area.configure(state='disabled')
            self.text_area.see(tk.END) # Auto-scroll
        # Asegurarse de que la actualización de la GUI se haga en el hilo principal
        self.text_area.after(0, append_message)

# Modelos de Agente
class ObjetoBinario:
    def __init__(self, nombre, num_categorias=5, bits_por_categoria=4):
        self.nombre = nombre
        self.num_categorias = num_categorias
        self.bits_por_categoria = bits_por_categoria
        # Asegurarse de que las categorías se inicializan correctamente si los parámetros cambian
        max_val = 2**bits_por_categoria - 1
        self.categorias = [bin(random.randint(0, max_val))[2:].zfill(bits_por_categoria) for _ in range(num_categorias)]

    def actualizar_categoria(self, indice, valor_str):
        if not (0 <= indice < self.num_categorias):
            raise ValueError(f"Índice de categoría inválido. Debe estar entre 0 y {self.num_categorias - 1}.")
        try:
            valor = int(valor_str)
        except ValueError:
             raise ValueError(f"El valor '{valor_str}' no es un entero válido.")

        max_val = 2**self.bits_por_categoria - 1
        if not (0 <= valor <= max_val):
            raise ValueError(f"Valor fuera de rango. Debe estar entre 0 y {max_val}.")
        self.categorias[indice] = bin(valor)[2:].zfill(self.bits_por_categoria)

    def obtener_binario(self):
        return "".join(self.categorias)

    def obtener_categorias(self):
        return self.categorias

    def to_dict(self):
        return {
            "nombre": self.nombre,
            "num_categorias": self.num_categorias,  # Guardar num_categorias
            "bits_por_categoria": self.bits_por_categoria, # Guardar bits_por_categoria
            "categorias": self.categorias
        }

    @classmethod
    def from_dict(cls, data):
        # Usa los valores guardados para la inicialización
        obj = cls(data["nombre"], data["num_categorias"], data["bits_por_categoria"])
        # Validar que las categorías cargadas coincidan con los parámetros
        if len(data["categorias"]) != obj.num_categorias:
             raise ValueError("Inconsistencia en el número de categorías cargadas.")
        for cat_str in data["categorias"]:
             if len(cat_str) != obj.bits_por_categoria:
                  raise ValueError("Inconsistencia en los bits por categoría cargados.")
        obj.categorias = data["categorias"]
        return obj

class EntornoSimulado:
    def __init__(self, objetos, recompensa_por_movimiento=0.0, recompensa_por_ajuste=0.1, penalidad_invalida=-0.05):
        if not objetos:
            raise ValueError("La lista de objetos no puede estar vacía para crear un entorno.")
        self.objetos = objetos
        self.estado_actual = 0 # Índice del objeto actual
        self.recompensa_por_movimiento = recompensa_por_movimiento
        self.recompensa_por_ajuste = recompensa_por_ajuste
        self.penalidad_invalida = penalidad_invalida
        # Calcular num_acciones_disponibles correctamente
        # Acciones: 0=derecha, 1=izquierda
        #           2=inc cat 0, 3=dec cat 0
        #           4=inc cat 1, 5=dec cat 1
        #           ...
        #           2 + 2*(num_cat-1) = inc cat (n-1)
        #           2 + 2*(num_cat-1) + 1 = dec cat (n-1)
        self.num_acciones_disponibles = 2 + 2 * self.objetos[0].num_categorias
        self.max_state_value = 2**(objetos[0].num_categorias * objetos[0].bits_por_categoria) - 1

    def obtener_estado(self):
        objeto_actual = self.objetos[self.estado_actual]
        # Normalizar el estado a [0, 1] podría ser útil para la red neuronal
        estado_numerico = int(objeto_actual.obtener_binario(), 2)
        # return estado_numerico / self.max_state_value # Normalizado
        return estado_numerico # Sin normalizar por ahora

    def ejecutar_accion(self, accion):
        objeto_actual = self.objetos[self.estado_actual]
        num_categorias = objeto_actual.num_categorias
        bits_por_categoria = objeto_actual.bits_por_categoria
        recompensa = 0

        if accion == 0:  # Mover a la derecha
            self.estado_actual = (self.estado_actual + 1) % len(self.objetos)
            recompensa = self.recompensa_por_movimiento
            logger.debug(f"Acción: Mover Derecha. Nuevo objeto índice: {self.estado_actual}")
        elif accion == 1:  # Mover a la izquierda
            self.estado_actual = (self.estado_actual - 1 + len(self.objetos)) % len(self.objetos)
            recompensa = self.recompensa_por_movimiento
            logger.debug(f"Acción: Mover Izquierda. Nuevo objeto índice: {self.estado_actual}")
        # Acciones de ajuste de categorías (2 por categoría)
        elif 2 <= accion < self.num_acciones_disponibles:
            indice_categoria = (accion - 2) // 2
            tipo_ajuste = (accion - 2) % 2 # 0 para incrementar, 1 para decrementar

            if 0 <= indice_categoria < num_categorias:
                valor_actual_str = objeto_actual.obtener_categorias()[indice_categoria]
                valor_actual = int(valor_actual_str, 2)
                max_val_categoria = 2**bits_por_categoria - 1

                if tipo_ajuste == 0: # Incrementar
                    nuevo_valor = min(max_val_categoria, valor_actual + 1)
                    if nuevo_valor > valor_actual:
                        objeto_actual.actualizar_categoria(indice_categoria, str(nuevo_valor))
                        recompensa = self.recompensa_por_ajuste
                        logger.debug(f"Acción: Inc Cat {indice_categoria}. Objeto {self.estado_actual}. Nuevo valor: {nuevo_valor}")
                    else:
                        recompensa = 0 # Ya estaba al máximo
                        logger.debug(f"Acción: Inc Cat {indice_categoria}. Objeto {self.estado_actual}. Valor ya al máximo ({valor_actual}).")
                else: # Decrementar
                    nuevo_valor = max(0, valor_actual - 1)
                    if nuevo_valor < valor_actual:
                        objeto_actual.actualizar_categoria(indice_categoria, str(nuevo_valor))
                        recompensa = self.recompensa_por_ajuste
                        logger.debug(f"Acción: Dec Cat {indice_categoria}. Objeto {self.estado_actual}. Nuevo valor: {nuevo_valor}")
                    else:
                        recompensa = 0 # Ya estaba al mínimo
                        logger.debug(f"Acción: Dec Cat {indice_categoria}. Objeto {self.estado_actual}. Valor ya al mínimo ({valor_actual}).")
            else:
                 # Esto no debería ocurrir si num_acciones_disponibles está bien calculado
                 recompensa = self.penalidad_invalida
                 logger.warning(f"Índice de categoría inválido calculado: {indice_categoria} para acción {accion}")

        else:
            recompensa = self.penalidad_invalida  # Penalización por acción inválida
            logger.warning(f"Acción inválida recibida: {accion}. Rango esperado: 0-{self.num_acciones_disponibles-1}")

        siguiente_estado_numerico = self.obtener_estado()
        # La tarea no parece tener un estado terminal claro, así que `done` siempre es False
        # A menos que lo definamos por número de pasos.
        done = False
        return siguiente_estado_numerico, recompensa, done # Devuelve done

    def obtener_texto_estado(self):
        if not self.objetos:
             return "Sin objetos"
        if self.estado_actual >= len(self.objetos):
             self.estado_actual = 0 # Asegurar índice válido
        objeto_actual = self.objetos[self.estado_actual]
        categorias_str = ", ".join([f"Cat{i}: {int(cat, 2)}" for i, cat in enumerate(objeto_actual.obtener_categorias())])
        return f"Obj {self.estado_actual}: {objeto_actual.nombre} [{categorias_str}]"

    def reset(self):
        self.estado_actual = 0
        # Podríamos aleatorizar el estado inicial de los objetos aquí si quisiéramos
        # for obj in self.objetos:
        #     obj.__init__(obj.nombre, obj.num_categorias, obj.bits_por_categoria) # Reinicia valores
        logger.info("Entorno reseteado.")
        return self.obtener_estado()

# Modelos de Redes Neuronales
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], dropout_rate=0.1): # Reducido dropout
        super(QNetwork, self).__init__()
        layers = []
        # Considerar una capa de Embedding si el state_dim es discreto y grande
        # if state_dim == 1: # Si el estado es un solo entero grande
        #     embedding_dim = 32 # Ejemplo
        #     # Necesitaríamos saber el número máximo de estados posibles (vocab_size)
        #     # max_state_value = ???
        #     # layers.append(nn.Embedding(max_state_value + 1, embedding_dim))
        #     # prev_dim = embedding_dim
        #     prev_dim = state_dim # Por ahora, tratarlo como continuo
        # else:
        prev_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.layers = nn.Sequential(*layers)
        logger.info(f"QNetwork creada: state_dim={state_dim}, action_dim={action_dim}, hidden={hidden_layers}")


    def forward(self, x):
        # Si x no es float, convertirlo
        if x.dtype != torch.float32:
            x = x.float()
        # Si el input es un escalar (int estado), convertirlo a tensor [1, 1] o [batch_size, 1]
        if len(x.shape) == 1:
             x = x.unsqueeze(-1) # Convierte [batch_size] a [batch_size, 1]
        elif len(x.shape) == 0:
             x = x.unsqueeze(0).unsqueeze(0) # Convierte escalar a [1, 1]

        # Si usamos Embedding, necesitaríamos x como Long y manejar la salida de Embedding
        # x = self.layers[0](x.long()) # Capa Embedding
        # x = x.view(x.size(0), -1) # Aplanar si es necesario
        # for layer in self.layers[1:]: # Resto de capas
        #    x = layer(x)
        # return x
        return self.layers(x) # Asumiendo entrada ya es [batch_size, state_dim]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], dropout_rate=0.1):
        super(ActorCritic, self).__init__()
        prev_dim = state_dim
        actor_layers = []
        for hidden_dim in hidden_layers:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.ReLU())
            actor_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        prev_dim = state_dim
        critic_layers = []
        for hidden_dim in hidden_layers:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))  # Output is a single value (state value)
        self.critic = nn.Sequential(*critic_layers)
        logger.info(f"ActorCritic creada: state_dim={state_dim}, action_dim={action_dim}, hidden={hidden_layers}")

    def forward(self, x):
         if x.dtype != torch.float32:
            x = x.float()
         if len(x.shape) == 1:
             x = x.unsqueeze(-1)
         elif len(x.shape) == 0:
             x = x.unsqueeze(0).unsqueeze(0)

         action_logits = self.actor(x)
         state_value = self.critic(x)
         # No aplicar Softmax aquí, Categorical lo prefiere en logits por estabilidad numérica
         return action_logits, state_value

# Agentes de Aprendizaje por Refuerzo
class QLearningAgent:
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], learning_rate=0.001, gamma=0.99, epsilon=0.99, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=10000, batch_size=64, tau=0.005): # Aumentado tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.q_network = QNetwork(state_dim, action_dim, hidden_layers)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_layers) # Target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with the same weights
        self.target_q_network.eval() # Set target network to evaluation mode
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=buffer_size)  # Replay buffer
        self.batch_size = batch_size
        self.tau = tau  # For soft updates of target network
        logger.info(f"QLearningAgent inicializado. LR={learning_rate}, Gamma={gamma}, Epsilon={epsilon}, Batch={batch_size}")

    def seleccionar_accion(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            logger.debug(f"Acción aleatoria (epsilon={self.epsilon:.3f}): {action}")
            return action
        else:
            with torch.no_grad():
                # El estado es un entero, convertirlo a tensor float [1, state_dim]
                # Asumiendo state_dim = 1
                state_tensor = torch.tensor([[state]], dtype=torch.float32)
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
                logger.debug(f"Acción greedy (Q-values: {q_values.numpy()}): {action}")
                return action

    def almacenar_experiencia(self, state, action, reward, next_state, done):
        # Guardar estados como números, no tensores
        self.memory.append((state, action, reward, next_state, done))
        logger.debug(f"Experiencia almacenada: S={state}, A={action}, R={reward:.2f}, S'={next_state}, Done={done}. Memoria: {len(self.memory)}")


    def aprender(self):
        if len(self.memory) < self.batch_size:
            #logger.debug(f"No hay suficientes muestras para aprender ({len(self.memory)}/{self.batch_size})")
            return None  # Not enough samples to learn from yet

        #logger.debug(f"Aprendiendo de batch de tamaño {self.batch_size}")
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir a tensores, asegurando la forma [batch_size, state_dim] para estados
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)  # [batch_size, 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1] (0.0 o 1.0)

        # Compute the target Q values using the target network
        with torch.no_grad():
            # Obtener los Q-values del siguiente estado desde la red target
            q_next_target = self.target_q_network(next_states)
            # Seleccionar el valor máximo de Q para el siguiente estado (max_a' Q_target(s', a'))
            max_q_next = q_next_target.max(dim=1, keepdim=True)[0]
            # Calcular el valor Q objetivo: r + gamma * max_a' Q_target(s', a') * (1 - done)
            target_q = rewards + (1.0 - dones) * self.gamma * max_q_next

        # Get current Q values from the main network for the actions taken
        # Seleccionar los Q-values de la red principal correspondientes a las acciones tomadas
        current_q = self.q_network(states).gather(1, actions)  # Q(s, a)

        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0) # Opcional: Gradiente clipping
        self.optimizer.step()

        # Soft update of target network
        self.soft_update_target_network()

        # Decay epsilon (outside learning check)
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # Moved decay to end of episode

        # logger.debug(f"Aprendizaje completado. Loss: {loss.item():.4f}")
        return loss.item()

    def decay_epsilon(self):
         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
         # logger.debug(f"Epsilon decaído a {self.epsilon:.4f}")

    def soft_update_target_network(self):
        #logger.debug("Actualizando red target (soft update)")
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def guardar_modelo(self, ruta):
        save_dict = {
            'agent_type': 'Q-Learning',
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_q_network.state_dict(), # Save target network
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_layers': self.hidden_layers,
            'memory': list(self.memory), # Guardar memoria como lista (puede ser grande)
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'buffer_size': self.memory.maxlen,
            'batch_size': self.batch_size,
            'tau': self.tau
        }
        torch.save(save_dict, ruta)
        logger.info(f"Modelo Q-Learning guardado en {ruta}")

    def cargar_modelo(self, ruta):
      try:
          checkpoint = torch.load(ruta)
          if checkpoint.get('agent_type') != 'Q-Learning':
               raise TypeError("El archivo no contiene un modelo Q-Learning")

          self.state_dim = checkpoint['state_dim']
          self.action_dim = checkpoint['action_dim']
          self.hidden_layers = checkpoint['hidden_layers']
          learning_rate = checkpoint.get('learning_rate', 0.001) # Default si no está guardado
          self.gamma = checkpoint['gamma']
          self.epsilon = checkpoint['epsilon']
          self.epsilon_decay = checkpoint.get('epsilon_decay', 0.995)
          self.epsilon_min = checkpoint.get('epsilon_min', 0.01)
          buffer_size = checkpoint.get('buffer_size', 10000)
          self.batch_size = checkpoint.get('batch_size', 64)
          self.tau = checkpoint.get('tau', 0.005)

          self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_layers)
          self.q_network.load_state_dict(checkpoint['model_state_dict'])

          self.target_q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_layers)
          self.target_q_network.load_state_dict(checkpoint['target_model_state_dict'])
          self.target_q_network.eval()

          self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

          # Cargar memoria si está guardada y no es demasiado grande
          saved_memory = checkpoint.get('memory')
          if saved_memory:
              self.memory = deque(saved_memory, maxlen=buffer_size)
          else:
              self.memory = deque(maxlen=buffer_size)

          logger.info(f"Modelo Q-Learning cargado desde {ruta}. Epsilon={self.epsilon:.3f}, Memoria={len(self.memory)}")
          return checkpoint # Devolver checkpoint para que la app actualice la UI si es necesario
      except FileNotFoundError:
           logger.error(f"Error al cargar modelo: Archivo no encontrado en {ruta}")
           raise
      except Exception as e:
           logger.error(f"Error al cargar modelo desde {ruta}: {e}")
           raise


class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_layers)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.learning_rate = learning_rate # Guardar para posible guardado/carga
        # Almacenamiento temporal para la trayectoria actual
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = [] # Necesitamos saber si un estado fue terminal
        logger.info(f"A2CAgent inicializado. LR={learning_rate}, Gamma={gamma}")

    def seleccionar_accion(self, state):
        # Convertir estado a tensor float [1, state_dim]
        state_tensor = torch.tensor([[state]], dtype=torch.float32)
        # Obtener logits y valor del estado actual
        logits, value = self.actor_critic(state_tensor)
        # Crear distribución de probabilidad sobre las acciones
        action_probs = Categorical(logits=logits) # Usar logits directamente
        # Muestrear una acción de la distribución
        action = action_probs.sample()

        # Guardar log_prob de la acción seleccionada y el valor del estado para el entrenamiento
        self.log_probs.append(action_probs.log_prob(action))
        self.values.append(value)

        logger.debug(f"Acción A2C (logits: {logits.detach().numpy()}): {action.item()}. Value: {value.item():.3f}")
        return action.item()

    def almacenar_transicion(self, reward, done):
        # Guarda la recompensa y si el paso fue terminal
        self.rewards.append(reward)
        self.dones.append(done)
        logger.debug(f"Transición A2C almacenada: R={reward:.2f}, Done={done}. Buffer actual: {len(self.rewards)}")


    def calcular_returns_y_advantages(self, next_value=0.0):
        # Calcula los retornos descontados G_t y las ventajas A_t = G_t - V(s_t)
        returns = []
        advantages = []
        R = next_value # Valor estimado del estado final (0 si es terminal)

        # Iterar hacia atrás sobre las recompensas y valores guardados
        for reward, done, value in zip(reversed(self.rewards), reversed(self.dones), reversed(self.values)):
            if done:
                R = 0.0 # Si el estado fue terminal, el retorno desde ahí es solo la recompensa
            # G_t = r_t + gamma * G_{t+1} (donde G_{t+1} es R calculado en la iteración anterior)
            R = reward + self.gamma * R
            returns.insert(0, R) # Añadir al principio

            # A_t = G_t - V(s_t)
            advantage = R - value.item() # value es un tensor [1,1], obtener el float
            advantages.insert(0, advantage)

        # Convertir a tensores
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalizar ventajas (opcional pero a menudo útil)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logger.debug(f"Retornos calculados: {returns.numpy()}")
        logger.debug(f"Ventajas calculadas (normalizadas): {advantages.numpy()}")
        return returns, advantages

    def aprender(self, next_state=None, done=False):
        # Se llama al final de una trayectoria (o episodio)
        if not self.rewards:
            logger.debug("Buffer de A2C vacío, no se puede aprender.")
            return None # No hay nada que aprender

        # 1. Calcular el valor del último estado (next_state) si la trayectoria no terminó (done=False)
        next_value = 0.0
        if next_state is not None and not done:
            with torch.no_grad():
                next_state_tensor = torch.tensor([[next_state]], dtype=torch.float32)
                _, next_value_tensor = self.actor_critic(next_state_tensor)
                next_value = next_value_tensor.item()
                logger.debug(f"Valor estimado del último estado (s'): {next_value:.3f}")

        # 2. Calcular Retornos (G_t) y Ventajas (A_t)
        returns, advantages = self.calcular_returns_y_advantages(next_value)

        # 3. Preparar tensores para el cálculo de pérdidas
        # values es una lista de tensores [1,1], convertirlos a [N]
        values_tensor = torch.cat(self.values).squeeze() # [N]
        # log_probs es una lista de tensores escalares, convertirlos a [N]
        log_probs_tensor = torch.stack(self.log_probs).squeeze() # [N]

        # 4. Calcular pérdidas
        # Pérdida del Actor (policy gradient): - mean(log_prob(a_t|s_t) * A_t)
        actor_loss = -(log_probs_tensor * advantages).mean()
        # Pérdida del Crítico (MSE): mean((G_t - V(s_t))^2)
        critic_loss = F.mse_loss(values_tensor, returns)

        # Pérdida total (a menudo se pondera la pérdida del crítico)
        total_loss = actor_loss + 0.5 * critic_loss # Coeficiente típico 0.5 para el crítico

        # 5. Optimización
        self.optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5) # Opcional: Gradiente clipping
        self.optimizer.step()

        # 6. Limpiar buffers para la próxima trayectoria
        self.reset()

        logger.debug(f"Aprendizaje A2C completado. Loss Total: {total_loss.item():.4f} (Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f})")
        return total_loss.item()

    def guardar_modelo(self, ruta):
        save_dict = {
            'agent_type': 'A2C',
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_layers': self.hidden_layers,
        }
        torch.save(save_dict, ruta)
        logger.info(f"Modelo A2C guardado en {ruta}")

    def cargar_modelo(self, ruta):
        try:
            checkpoint = torch.load(ruta)
            if checkpoint.get('agent_type') != 'A2C':
                raise TypeError("El archivo no contiene un modelo A2C")

            self.state_dim = checkpoint['state_dim']
            self.action_dim = checkpoint['action_dim']
            self.hidden_layers = checkpoint['hidden_layers']
            self.learning_rate = checkpoint.get('learning_rate', 0.001)
            self.gamma = checkpoint['gamma']

            self.actor_critic = ActorCritic(self.state_dim, self.action_dim, self.hidden_layers)
            self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            self.actor_critic.eval() # Poner en modo evaluación inicialmente

            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.reset() # Limpiar buffers
            logger.info(f"Modelo A2C cargado desde {ruta}.")
            return checkpoint # Devolver checkpoint para que la app actualice la UI si es necesario
        except FileNotFoundError:
           logger.error(f"Error al cargar modelo A2C: Archivo no encontrado en {ruta}")
           raise
        except Exception as e:
           logger.error(f"Error al cargar modelo A2C desde {ruta}: {e}")
           raise

    def reset(self):
        # Limpiar los buffers de la trayectoria
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        logger.debug("Buffers de trayectoria A2C reseteados.")

# Clase principal de la aplicación
class QuantumAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Agent Simulator")
        self.root.geometry("1200x850") # Un poco más alto

        # Variables de control
        self.entorno = None
        self.agente = None
        self.agente_tipo = tk.StringVar(value="Q-Learning")
        self.entrenamiento_activo = False
        self.objetos = []
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.historial_epsilon = []
        self.ultimo_episodio = 0
        self.max_pasos_por_episodio = 200 # Límite de pasos por episodio

        # Configurar estructura de la interfaz
        self.crear_estructura_ui()

        # Inicializar objetos por defecto
        self.crear_objetos_predeterminados() # Esto también crea el entorno

        # Actualizar la información de estado inicial
        self.actualizar_info_estado()
        self.actualizar_vista_objetos()

        # Configurar cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        logger.info("Aplicación inicializada.")

    def crear_estructura_ui(self):
        # Panel principal
        panel_principal = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panel_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel izquierdo (controles)
        panel_izquierdo = ttk.Frame(panel_principal, width=400) # Darle un ancho inicial
        panel_izquierdo.pack_propagate(False) # Evitar que se encoja
        panel_principal.add(panel_izquierdo, weight=1)

        # Panel derecho (visualización)
        panel_derecho = ttk.Frame(panel_principal, width=800)
        panel_derecho.pack_propagate(False)
        panel_principal.add(panel_derecho, weight=3)

        # Configuración de los paneles
        self.configurar_panel_izquierdo(panel_izquierdo)
        self.configurar_panel_derecho(panel_derecho)

    def configurar_panel_izquierdo(self, panel):
        # Frame para la información del entorno
        frame_entorno = ttk.LabelFrame(panel, text="Entorno")
        frame_entorno.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Información de estado actual
        ttk.Label(frame_entorno, text="Estado actual:").pack(anchor=tk.W, padx=5, pady=2)
        self.lbl_estado_actual = ttk.Label(frame_entorno, text="No iniciado", wraplength=380) # Wrap text
        self.lbl_estado_actual.pack(anchor=tk.W, padx=5, pady=2)

        # Lista de objetos
        frame_objetos_list = ttk.LabelFrame(panel, text="Objetos del Entorno")
        frame_objetos_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_tree = ttk.Frame(frame_objetos_list)
        frame_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(frame_tree)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_objetos = ttk.Treeview(frame_tree, columns=("Nombre", "Categorías"), show="headings", height=5)
        self.tree_objetos.heading("Nombre", text="Nombre")
        self.tree_objetos.heading("Categorías", text="Valores Categorías")
        self.tree_objetos.column("Nombre", width=100, anchor=tk.W)
        self.tree_objetos.column("Categorías", width=200, anchor=tk.W)
        self.tree_objetos.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.tree_objetos.yview)
        self.tree_objetos.config(yscrollcommand=scrollbar.set)

        # Botones para gestionar objetos
        frame_botones_objetos = ttk.Frame(frame_objetos_list)
        frame_botones_objetos.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(frame_botones_objetos, text="Añadir", command=self.mostrar_dialogo_nuevo_objeto).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_objetos, text="Editar", command=self.editar_objeto_seleccionado).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_objetos, text="Eliminar", command=self.eliminar_objeto_seleccionado).pack(side=tk.LEFT, padx=2)

        # Frame para configuración del agente
        frame_agente = ttk.LabelFrame(panel, text="Configuración del Agente")
        frame_agente.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Tipo de agente
        frame_tipo_agente = ttk.Frame(frame_agente)
        frame_tipo_agente.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frame_tipo_agente, text="Tipo:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(frame_tipo_agente, text="Q-Learning", variable=self.agente_tipo, value="Q-Learning", command=self.toggle_epsilon_params).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(frame_tipo_agente, text="A2C", variable=self.agente_tipo, value="A2C", command=self.toggle_epsilon_params).pack(side=tk.LEFT, padx=5)

        # Parámetros de entrenamiento
        frame_params = ttk.Frame(frame_agente)
        frame_params.pack(fill=tk.X, padx=5, pady=2)

        # Fila 1: LR, Gamma
        ttk.Label(frame_params, text="LR:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=1)
        self.entry_lr = ttk.Entry(frame_params, width=7)
        self.entry_lr.insert(0, "0.001")
        self.entry_lr.grid(row=0, column=1, sticky=tk.W, padx=2, pady=1)

        ttk.Label(frame_params, text="Gamma:").grid(row=0, column=2, sticky=tk.W, padx=2, pady=1)
        self.entry_gamma = ttk.Entry(frame_params, width=7)
        self.entry_gamma.insert(0, "0.99")
        self.entry_gamma.grid(row=0, column=3, sticky=tk.W, padx=2, pady=1)

        # Fila 2: Epsilon (solo Q-Learning)
        self.lbl_epsilon = ttk.Label(frame_params, text="Epsilon:")
        self.lbl_epsilon.grid(row=1, column=0, sticky=tk.W, padx=2, pady=1)
        self.entry_epsilon = ttk.Entry(frame_params, width=7)
        self.entry_epsilon.insert(0, "0.99")
        self.entry_epsilon.grid(row=1, column=1, sticky=tk.W, padx=2, pady=1)

        self.lbl_epsilon_decay = ttk.Label(frame_params, text="Eps Decay:")
        self.lbl_epsilon_decay.grid(row=1, column=2, sticky=tk.W, padx=2, pady=1)
        self.entry_epsilon_decay = ttk.Entry(frame_params, width=7)
        self.entry_epsilon_decay.insert(0, "0.995")
        self.entry_epsilon_decay.grid(row=1, column=3, sticky=tk.W, padx=2, pady=1)

        # Arquitectura de red
        frame_red = ttk.Frame(frame_agente)
        frame_red.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frame_red, text="Capas Ocultas (ej: 128,64):").pack(side=tk.LEFT, padx=5)
        self.entry_hidden_layers = ttk.Entry(frame_red, width=15)
        self.entry_hidden_layers.insert(0, "128, 64")
        self.entry_hidden_layers.pack(side=tk.LEFT, padx=5)

        # Botones de entrenamiento
        frame_botones_entrenamiento = ttk.Frame(frame_agente)
        frame_botones_entrenamiento.pack(fill=tk.X, padx=5, pady=5)

        self.btn_iniciar = ttk.Button(frame_botones_entrenamiento, text="Iniciar", command=self.iniciar_entrenamiento)
        self.btn_iniciar.pack(side=tk.LEFT, padx=2)

        self.btn_detener = ttk.Button(frame_botones_entrenamiento, text="Detener", command=self.detener_entrenamiento, state=tk.DISABLED)
        self.btn_detener.pack(side=tk.LEFT, padx=2)

        ttk.Button(frame_botones_entrenamiento, text="Reiniciar Entorno", command=self.reiniciar_entorno).pack(side=tk.LEFT, padx=2)

        # Frame para acciones manuales
        frame_acciones = ttk.LabelFrame(panel, text="Acciones Manuales")
        frame_acciones.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Botones de acciones
        frame_botones_acciones = ttk.Frame(frame_acciones)
        frame_botones_acciones.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(frame_botones_acciones, text="◀ Izq (1)", command=lambda: self.ejecutar_accion_manual(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_acciones, text="▶ Der (0)", command=lambda: self.ejecutar_accion_manual(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_acciones, text="▲ Inc C0 (2)", command=lambda: self.ejecutar_accion_manual(2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_acciones, text="▼ Dec C0 (3)", command=lambda: self.ejecutar_accion_manual(3)).pack(side=tk.LEFT, padx=2)
        # Añadir más botones si es necesario o un modo para seleccionar categoría y acción

        # Frame para guardar/cargar
        frame_guardar_cargar = ttk.Frame(panel)
        frame_guardar_cargar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(frame_guardar_cargar, text="Guardar Modelo", command=self.guardar_modelo).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(frame_guardar_cargar, text="Cargar Modelo", command=self.cargar_modelo).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(frame_guardar_cargar, text="Exportar Entorno", command=self.exportar_entorno).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(frame_guardar_cargar, text="Importar Entorno", command=self.importar_entorno).pack(side=tk.LEFT, padx=2, pady=2)

        # Ajustar visibilidad inicial de parámetros Epsilon
        self.toggle_epsilon_params()

    def configurar_panel_derecho(self, panel):
        # Notebook para pestañas
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña de visualización de entrenamiento
        tab_entrenamiento = ttk.Frame(notebook)
        notebook.add(tab_entrenamiento, text="Entrenamiento")

        # Frame para gráficos
        frame_graficos = ttk.Frame(tab_entrenamiento)
        frame_graficos.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Crear figura para gráficos
        self.fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True) # Ajustar tamaño
        self.ax1, self.ax2, self.ax3 = axes
        self.fig.tight_layout(pad=4.0) # Más padding

        # Gráfico de recompensas
        self.ax1.set_title("Recompensa Acumulada por Episodio")
        self.ax1.set_ylabel("Recompensa Total")
        self.ax1.grid(True)
        self.line_reward, = self.ax1.plot([], [], 'b-', label='Recompensa')
        self.ax1.legend(loc='upper left')

        # Gráfico de pérdidas
        self.ax2.set_title("Pérdida Media por Episodio")
        self.ax2.set_ylabel("Pérdida")
        self.ax2.grid(True)
        self.line_loss, = self.ax2.plot([], [], 'r-', label='Pérdida')
        self.ax2.legend(loc='upper left')

        # Gráfico de epsilon (para Q-Learning)
        self.ax3.set_title("Epsilon por Episodio (Q-Learning)")
        self.ax3.set_xlabel("Episodio")
        self.ax3.set_ylabel("Epsilon")
        self.ax3.grid(True)
        self.line_epsilon, = self.ax3.plot([], [], 'g-', label='Epsilon')
        self.ax3.legend(loc='upper left')
        self.ax3.set_ylim(0, 1.1) # Rango fijo para Epsilon

        # Crear canvas para mostrar los gráficos
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graficos)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # Pestaña de log
        tab_log = ttk.Frame(notebook)
        notebook.add(tab_log, text="Log")

        # Área de texto para mostrar logs
        self.txt_log = scrolledtext.ScrolledText(tab_log, state='disabled', wrap=tk.WORD, height=10) # Read-only
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configurar handler para mostrar logs en UI
        self.log_handler = LogTextHandler(self.txt_log)
        self.log_handler.setLevel(logging.INFO) # Mostrar INFO y superior en UI
        # Usar un formato más simple para la UI
        log_format_ui = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.log_handler.setFormatter(log_format_ui)
        logger.addHandler(self.log_handler) # Añadir al logger global

    def toggle_epsilon_params(self):
        # Habilita/deshabilita los campos de Epsilon según el agente seleccionado
        if self.agente_tipo.get() == "Q-Learning":
            self.lbl_epsilon.config(state=tk.NORMAL)
            self.entry_epsilon.config(state=tk.NORMAL)
            self.lbl_epsilon_decay.config(state=tk.NORMAL)
            self.entry_epsilon_decay.config(state=tk.NORMAL)
            # Mostrar gráfico de Epsilon
            self.ax3.set_visible(True)

        else: # A2C
            self.lbl_epsilon.config(state=tk.DISABLED)
            self.entry_epsilon.config(state=tk.DISABLED)
            self.lbl_epsilon_decay.config(state=tk.DISABLED)
            self.entry_epsilon_decay.config(state=tk.DISABLED)
            # Ocultar gráfico de Epsilon
            self.ax3.set_visible(False)
        # Redibujar canvas si la visibilidad cambió
        self.canvas.draw_idle()

    def crear_objetos_predeterminados(self):
        logger.info("Creando objetos predeterminados.")
        # Crear algunos objetos por defecto
        # Asegurarse de que tengan la misma estructura (num_cat, bits_cat) para el entorno actual
        num_cat_default = 3
        bits_cat_default = 4
        objeto1 = ObjetoBinario("Sensor A", num_categorias=num_cat_default, bits_por_categoria=bits_cat_default)
        objeto2 = ObjetoBinario("Actuador B", num_categorias=num_cat_default, bits_por_categoria=bits_cat_default)
        objeto3 = ObjetoBinario("Estado C", num_categorias=num_cat_default, bits_por_categoria=bits_cat_default)

        self.objetos = [objeto1, objeto2, objeto3]

        # Crear entorno con estos objetos
        try:
            self.entorno = EntornoSimulado(self.objetos)
            logger.info(f"Entorno creado con {len(self.objetos)} objetos. Acciones disponibles: {self.entorno.num_acciones_disponibles}")
            # No es necesario actualizar la vista aquí, se hace en __init__
        except ValueError as e:
            logger.error(f"Error al crear el entorno inicial: {e}")
            messagebox.showerror("Error de Entorno", f"No se pudo crear el entorno: {e}")
            self.entorno = None # Asegurarse de que el entorno es None

    def actualizar_vista_objetos(self):
        # Limpiar vista actual
        for item in self.tree_objetos.get_children():
            self.tree_objetos.delete(item)

        # Rellenar con los objetos actuales
        for objeto in self.objetos:
            try:
                categorias_vals = [str(int(cat, 2)) for cat in objeto.obtener_categorias()]
                categorias_str = ", ".join(categorias_vals)
                self.tree_objetos.insert("", tk.END, values=(objeto.nombre, categorias_str))
            except ValueError as e:
                 logger.error(f"Error al mostrar objeto {objeto.nombre}: {e}. Categorías: {objeto.categorias}")
                 self.tree_objetos.insert("", tk.END, values=(objeto.nombre, "Error de formato"))
            except Exception as e:
                 logger.error(f"Error inesperado al mostrar objeto {objeto.nombre}: {e}")
                 self.tree_objetos.insert("", tk.END, values=(objeto.nombre, "Error"))


    def mostrar_dialogo_nuevo_objeto(self):
        # Crear diálogo para nuevo objeto
        dialogo = tk.Toplevel(self.root)
        dialogo.title("Nuevo Objeto")
        dialogo.geometry("350x200") # Ajustar tamaño
        dialogo.transient(self.root)
        dialogo.grab_set()
        dialogo.resizable(False, False)

        # Si ya existen objetos, usar sus parámetros como predeterminados
        default_num_cat = 3
        default_bits = 4
        if self.objetos:
             default_num_cat = self.objetos[0].num_categorias
             default_bits = self.objetos[0].bits_por_categoria
             can_change_structure = False # No permitir cambiar si ya hay objetos
        else:
             can_change_structure = True # Permitir si es el primer objeto

        # Campos del formulario
        frame_form = ttk.Frame(dialogo, padding=10)
        frame_form.pack(expand=True, fill=tk.BOTH)

        ttk.Label(frame_form, text="Nombre:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry_nombre = ttk.Entry(frame_form, width=25)
        entry_nombre.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Label(frame_form, text="Número de categorías:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        entry_num_cat = ttk.Entry(frame_form, width=10)
        entry_num_cat.insert(0, str(default_num_cat))
        entry_num_cat.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        entry_num_cat.config(state=tk.NORMAL if can_change_structure else tk.DISABLED)

        ttk.Label(frame_form, text="Bits por categoría:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        entry_bits = ttk.Entry(frame_form, width=10)
        entry_bits.insert(0, str(default_bits))
        entry_bits.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        entry_bits.config(state=tk.NORMAL if can_change_structure else tk.DISABLED)

        if not can_change_structure:
            ttk.Label(frame_form, text="(Estructura fija por objetos existentes)", foreground="grey").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5)


        # Botones
        frame_botones = ttk.Frame(dialogo)
        frame_botones.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        frame_botones.columnconfigure(0, weight=1)
        frame_botones.columnconfigure(1, weight=1)

        def guardar_objeto():
            try:
                nombre = entry_nombre.get().strip()
                if not nombre:
                    messagebox.showerror("Error", "El nombre no puede estar vacío", parent=dialogo)
                    return

                num_cat = int(entry_num_cat.get())
                bits = int(entry_bits.get())

                if num_cat <= 0 or bits <= 0:
                    messagebox.showerror("Error", "Los valores numéricos deben ser positivos", parent=dialogo)
                    return
                if bits > 16: # Límite práctico
                    messagebox.showwarning("Advertencia", "Un número alto de bits puede hacer el espacio de estados muy grande.", parent=dialogo)

                # Verificar consistencia si no se podía cambiar la estructura
                if not can_change_structure and (num_cat != default_num_cat or bits != default_bits):
                     messagebox.showerror("Error", f"La estructura debe ser {default_num_cat} categorías con {default_bits} bits cada una.", parent=dialogo)
                     # Resetear valores
                     entry_num_cat.delete(0, tk.END)
                     entry_num_cat.insert(0, str(default_num_cat))
                     entry_bits.delete(0, tk.END)
                     entry_bits.insert(0, str(default_bits))
                     return


                # Crear nuevo objeto
                nuevo_objeto = ObjetoBinario(nombre, num_cat, bits)
                self.objetos.append(nuevo_objeto)
                logger.info(f"Objeto '{nombre}' añadido con {num_cat} cats, {bits} bits/cat.")

                # Actualizar vista
                self.actualizar_vista_objetos()

                # Recrear entorno si es el primer objeto o si la estructura cambió (aunque lo impedimos)
                if self.entorno is None or len(self.objetos) == 1:
                    self.reiniciar_entorno_completo() # Recrea entorno y resetea agente
                else:
                    # Solo actualizar la lista de objetos en el entorno existente
                    self.entorno.objetos = self.objetos
                    logger.info(f"Lista de objetos del entorno actualizada. Total: {len(self.objetos)}")
                    self.actualizar_info_estado() # Actualizar UI

                dialogo.destroy()

            except ValueError as e:
                messagebox.showerror("Error", f"Valor inválido: {e}", parent=dialogo)
            except Exception as e:
                 logger.exception("Error inesperado al guardar nuevo objeto.")
                 messagebox.showerror("Error", f"Error inesperado: {e}", parent=dialogo)

        btn_guardar = ttk.Button(frame_botones, text="Guardar", command=guardar_objeto)
        btn_guardar.grid(row=0, column=0, padx=5, sticky=tk.E)
        btn_cancelar = ttk.Button(frame_botones, text="Cancelar", command=dialogo.destroy)
        btn_cancelar.grid(row=0, column=1, padx=5, sticky=tk.W)

        dialogo.wait_window()


    def editar_objeto_seleccionado(self):
        seleccion = self.tree_objetos.selection()
        if not seleccion:
            messagebox.showinfo("Información", "Seleccione un objeto para editar")
            return

        item_id = seleccion[0]
        indice = self.tree_objetos.index(item_id)

        if not (0 <= indice < len(self.objetos)):
             logger.error(f"Índice de objeto inválido al editar: {indice}")
             messagebox.showerror("Error", "No se pudo encontrar el objeto seleccionado.")
             return

        objeto = self.objetos[indice]

        # Crear diálogo para editar
        dialogo = tk.Toplevel(self.root)
        dialogo.title(f"Editar Objeto")
        dialogo.geometry("400x450") # Más alto para categorías
        dialogo.transient(self.root)
        dialogo.grab_set()
        dialogo.resizable(False, True) # Permitir redimensionar verticalmente

        # Frame principal con padding
        main_frame = ttk.Frame(dialogo, padding=10)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Campos básicos (Nombre)
        frame_basic = ttk.Frame(main_frame)
        frame_basic.pack(fill=tk.X, pady=5)
        ttk.Label(frame_basic, text="Nombre:").pack(side=tk.LEFT, padx=5)
        entry_nombre = ttk.Entry(frame_basic, width=30)
        entry_nombre.insert(0, objeto.nombre)
        entry_nombre.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Mostrar estructura (no editable)
        frame_structure = ttk.Frame(main_frame)
        frame_structure.pack(fill=tk.X, pady=5)
        ttk.Label(frame_structure, text=f"Categorías: {objeto.num_categorias}").pack(side=tk.LEFT, padx=5)
        ttk.Label(frame_structure, text=f"Bits por cat: {objeto.bits_por_categoria}").pack(side=tk.LEFT, padx=15)
        max_val_cat = 2**objeto.bits_por_categoria - 1
        ttk.Label(frame_structure, text=f"(Valores: 0-{max_val_cat})", foreground="grey").pack(side=tk.LEFT, padx=5)


        # Frame para valores de categorías con scroll
        frame_categorias_outer = ttk.LabelFrame(main_frame, text="Valores de Categorías")
        frame_categorias_outer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        cat_canvas = tk.Canvas(frame_categorias_outer)
        scrollbar_cat = ttk.Scrollbar(frame_categorias_outer, orient="vertical", command=cat_canvas.yview)
        frame_categorias = ttk.Frame(cat_canvas) # Frame interior para contenido

        frame_categorias.bind(
            "<Configure>",
            lambda e: cat_canvas.configure(scrollregion=cat_canvas.bbox("all"))
        )

        cat_canvas.create_window((0, 0), window=frame_categorias, anchor="nw")
        cat_canvas.configure(yscrollcommand=scrollbar_cat.set)

        cat_canvas.pack(side="left", fill="both", expand=True)
        scrollbar_cat.pack(side="right", fill="y")


        # Campos para cada categoría dentro del frame scrolleable
        entries_categoria = []
        for i, categoria_bin in enumerate(objeto.obtener_categorias()):
            try:
                 valor_int = int(categoria_bin, 2)
            except ValueError:
                 valor_int = 0 # Default en caso de error
                 logger.warning(f"Valor binario inválido '{categoria_bin}' en objeto {objeto.nombre}, cat {i}. Usando 0.")

            ttk.Label(frame_categorias, text=f"Categoría {i}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
            entry_cat = ttk.Entry(frame_categorias, width=10)
            entry_cat.insert(0, str(valor_int))
            entry_cat.grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)
            entries_categoria.append(entry_cat)

        # Botones
        frame_botones = ttk.Frame(main_frame)
        frame_botones.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        frame_botones.columnconfigure(0, weight=1)
        frame_botones.columnconfigure(1, weight=1)

        def guardar_cambios():
            try:
                nuevo_nombre = entry_nombre.get().strip()
                if not nuevo_nombre:
                     messagebox.showerror("Error", "El nombre no puede estar vacío", parent=dialogo)
                     return
                objeto.nombre = nuevo_nombre

                # Actualizar valores de categorías
                nuevos_valores_bin = []
                for i, entry in enumerate(entries_categoria):
                    valor_str = entry.get().strip()
                    try:
                         # Validar y actualizar usando el método del objeto
                         objeto.actualizar_categoria(i, valor_str)
                         nuevos_valores_bin.append(objeto.obtener_categorias()[i]) # Guardar el valor actualizado
                    except ValueError as e:
                         messagebox.showerror("Error", f"Valor inválido para Categoría {i}: {e}\n(Rango: 0-{max_val_cat})", parent=dialogo)
                         return # Detener el guardado si hay un error

                logger.info(f"Objeto '{objeto.nombre}' (índice {indice}) editado. Nuevos valores: {nuevos_valores_bin}")

                # Actualizar vista de árbol
                categorias_display = ", ".join([str(int(cat, 2)) for cat in nuevos_valores_bin])
                self.tree_objetos.item(item_id, values=(objeto.nombre, categorias_display))

                # Actualizar estado actual en la UI si el objeto editado es el actual
                if self.entorno and self.entorno.estado_actual == indice:
                    self.actualizar_info_estado()

                dialogo.destroy()

            except Exception as e:
                 logger.exception("Error inesperado al guardar cambios del objeto.")
                 messagebox.showerror("Error", f"Error inesperado: {e}", parent=dialogo)

        btn_guardar = ttk.Button(frame_botones, text="Guardar", command=guardar_cambios)
        btn_guardar.grid(row=0, column=0, padx=5, sticky=tk.E)
        btn_cancelar = ttk.Button(frame_botones, text="Cancelar", command=dialogo.destroy)
        btn_cancelar.grid(row=0, column=1, padx=5, sticky=tk.W)

        dialogo.wait_window()


    def eliminar_objeto_seleccionado(self):
        seleccion = self.tree_objetos.selection()
        if not seleccion:
            messagebox.showinfo("Información", "Seleccione un objeto para eliminar")
            return

        item_id = seleccion[0]
        indice = self.tree_objetos.index(item_id)

        if not (0 <= indice < len(self.objetos)):
             logger.error(f"Índice de objeto inválido al eliminar: {indice}")
             messagebox.showerror("Error", "No se pudo encontrar el objeto seleccionado.")
             return

        objeto_a_eliminar = self.objetos[indice]

        # Confirmar eliminación
        if messagebox.askyesno("Confirmar", f"¿Seguro que desea eliminar el objeto '{objeto_a_eliminar.nombre}'?"):
            # Detener entrenamiento si está activo
            if self.entrenamiento_activo:
                 self.detener_entrenamiento()
                 messagebox.showinfo("Información", "El entrenamiento ha sido detenido.")

            # Eliminar objeto
            del self.objetos[indice]
            logger.info(f"Objeto '{objeto_a_eliminar.nombre}' (índice {indice}) eliminado.")

            # Si no quedan objetos, añadir uno por defecto para evitar errores
            if not self.objetos:
                logger.warning("Todos los objetos fueron eliminados. Añadiendo uno por defecto.")
                self.objetos.append(ObjetoBinario("Objeto Default", 3, 4))
                messagebox.showinfo("Información", "Se ha añadido un objeto por defecto ya que el entorno no puede estar vacío.")

            # Recrear/Reiniciar entorno y agente
            self.reiniciar_entorno_completo()

            # Actualizar vista (ya lo hace reiniciar_entorno_completo)
            #self.actualizar_vista_objetos()
            #self.actualizar_info_estado()


    def actualizar_info_estado(self):
        if self.entorno:
            try:
                texto_estado = self.entorno.obtener_texto_estado()
                self.lbl_estado_actual.config(text=texto_estado)
                # No loguear esto cada vez, es demasiado frecuente. Loguear cambios importantes.
            except Exception as e:
                 logger.error(f"Error al obtener/mostrar estado del entorno: {e}")
                 self.lbl_estado_actual.config(text="Error obteniendo estado")
        else:
            self.lbl_estado_actual.config(text="Entorno no disponible")

    def iniciar_entrenamiento(self):
        if not self.objetos:
             messagebox.showerror("Error", "No hay objetos en el entorno. Añada al menos uno.")
             logger.error("Intento de iniciar entrenamiento sin objetos.")
             return
        if not self.entorno:
             # Intentar recrear el entorno
             logger.warning("Entorno no existe al iniciar entrenamiento. Intentando recrear.")
             try:
                  self.entorno = EntornoSimulado(self.objetos)
                  logger.info("Entorno recreado con éxito.")
             except ValueError as e:
                  messagebox.showerror("Error", f"No se pudo crear el entorno: {e}")
                  logger.error(f"Fallo al recrear el entorno: {e}")
                  return

        if self.entrenamiento_activo:
            logger.warning("Intento de iniciar entrenamiento cuando ya está activo.")
            return

        # Validar y obtener hiperparámetros
        try:
            learning_rate = float(self.entry_lr.get())
            gamma = float(self.entry_gamma.get())

            if not (0 < learning_rate < 1):
                 raise ValueError("Learning Rate debe estar entre 0 y 1")
            if not (0 <= gamma <= 1):
                 raise ValueError("Gamma debe estar entre 0 y 1")

            epsilon = 0.0
            epsilon_decay = 0.0
            if self.agente_tipo.get() == "Q-Learning":
                epsilon = float(self.entry_epsilon.get())
                epsilon_decay = float(self.entry_epsilon_decay.get())
                if not (0 <= epsilon <= 1):
                     raise ValueError("Epsilon inicial debe estar entre 0 y 1")
                if not (0 < epsilon_decay <= 1):
                     raise ValueError("Epsilon decay debe estar entre 0 y 1")

            # Arquitectura de red
            hidden_layers_str = self.entry_hidden_layers.get()
            if not re.match(r"^\s*\d+(\s*,\s*\d+)*\s*$", hidden_layers_str):
                 raise ValueError("Formato de capas ocultas inválido (ej: 128, 64)")
            hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()]
            if not hidden_layers:
                 raise ValueError("Debe especificar al menos una capa oculta.")

            # Estado y Dimensiones de Acción
            # state_dim = self.objetos[0].num_categorias * self.objetos[0].bits_por_categoria # Si el estado fuera el vector binario
            state_dim = 1 # Si el estado es el entero combinado
            action_dim = self.entorno.num_acciones_disponibles

            logger.info(f"Iniciando entrenamiento con: Tipo={self.agente_tipo.get()}, LR={learning_rate}, Gamma={gamma}, Hidden={hidden_layers}")
            if self.agente_tipo.get() == "Q-Learning":
                logger.info(f"Epsilon={epsilon}, EpsDecay={epsilon_decay}")

            # Crear o reconfigurar agente (si no existe o cambió el tipo/estructura)
            if self.agente is None or \
               (self.agente_tipo.get() == "Q-Learning" and not isinstance(self.agente, QLearningAgent)) or \
               (self.agente_tipo.get() == "A2C" and not isinstance(self.agente, A2CAgent)) or \
               self.agente.state_dim != state_dim or \
               self.agente.action_dim != action_dim or \
               self.agente.hidden_layers != hidden_layers:
                logger.info("Creando nueva instancia del agente.")
                if self.agente_tipo.get() == "Q-Learning":
                    self.agente = QLearningAgent(state_dim, action_dim, hidden_layers, learning_rate, gamma, epsilon, epsilon_decay)
                else: # A2C
                    self.agente = A2CAgent(state_dim, action_dim, hidden_layers, learning_rate, gamma)
            else:
                 # Actualizar parámetros del agente existente
                 logger.info("Actualizando parámetros del agente existente.")
                 self.agente.gamma = gamma
                 self.agente.optimizer.param_groups[0]['lr'] = learning_rate
                 if isinstance(self.agente, QLearningAgent):
                      self.agente.epsilon = epsilon # Reiniciar epsilon al valor inicial? O continuar? Decidimos reiniciar.
                      self.agente.epsilon_decay = epsilon_decay
                      self.agente.epsilon_min = 0.01 # Hardcoded min value

            # Resetear historial y gráficos
            self.historial_recompensas = []
            self.historial_perdidas = []
            self.historial_epsilon = []
            self.ultimo_episodio = 0
            self.actualizar_graficos()

            # Cambiar estado de botones y flag
            self.entrenamiento_activo = True
            self.btn_iniciar.config(state=tk.DISABLED)
            self.btn_detener.config(state=tk.NORMAL)
            self._bloquear_controles_entorno(True) # Bloquear edición de entorno

            logger.info("***** Entrenamiento iniciado *****")
            # Iniciar ciclo de entrenamiento (usando after para no bloquear GUI)
            self.root.after(10, self.ciclo_entrenamiento)

        except ValueError as e:
            messagebox.showerror("Error de Parámetros", f"Valor inválido: {e}")
            logger.error(f"Error en parámetros de entrenamiento: {e}")
            # Asegurarse de que los botones estén en estado correcto si falla
            self.btn_iniciar.config(state=tk.NORMAL)
            self.btn_detener.config(state=tk.DISABLED)
            self._bloquear_controles_entorno(False)
        except Exception as e:
            logger.exception("Error inesperado al iniciar entrenamiento.")
            messagebox.showerror("Error", f"Error inesperado: {e}")
            self.btn_iniciar.config(state=tk.NORMAL)
            self.btn_detener.config(state=tk.DISABLED)
            self._bloquear_controles_entorno(False)

    def ciclo_entrenamiento(self):
        # Condición de parada
        if not self.entrenamiento_activo:
            logger.info("Ciclo de entrenamiento detenido.")
            return

        try:
            # Realizar un episodio completo
            estado = self.entorno.reset()
            if isinstance(self.agente, A2CAgent):
                 self.agente.reset() # Limpiar buffers A2C al inicio del episodio

            recompensa_total_episodio = 0
            perdidas_episodio = []
            done = False
            paso = 0

            while paso < self.max_pasos_por_episodio and not done:
                # Seleccionar acción
                accion = self.agente.seleccionar_accion(estado)

                # Ejecutar acción en el entorno
                siguiente_estado, recompensa, done_env = self.entorno.ejecutar_accion(accion)
                # Aquí 'done' se basa en max_pasos, no en done_env que siempre es False
                paso += 1
                done = (paso == self.max_pasos_por_episodio)

                # Almacenar experiencia/transición
                if isinstance(self.agente, QLearningAgent):
                    self.agente.almacenar_experiencia(estado, accion, recompensa, siguiente_estado, done)
                    # Aprender (si hay suficientes datos en memoria)
                    loss = self.agente.aprender()
                    if loss is not None:
                        perdidas_episodio.append(loss)
                elif isinstance(self.agente, A2CAgent):
                    self.agente.almacenar_transicion(recompensa, done) # Guardar r_t, done_t

                # Actualizar estado
                estado = siguiente_estado
                recompensa_total_episodio += recompensa

                # Si el episodio termina (por pasos), salir del bucle while
                if done:
                    break

            # Fin del episodio
            self.ultimo_episodio += 1

            # Si es A2C, aprender al final del episodio
            if isinstance(self.agente, A2CAgent):
                 loss = self.agente.aprender(next_state=estado, done=done) # Pasar último estado y si terminó
                 if loss is not None:
                      perdidas_episodio.append(loss)

            # Calcular pérdida media y almacenar historial
            perdida_media = np.mean(perdidas_episodio) if perdidas_episodio else 0
            self.historial_recompensas.append(recompensa_total_episodio)
            self.historial_perdidas.append(perdida_media)

            # Decaer epsilon para Q-Learning al final del episodio
            if isinstance(self.agente, QLearningAgent):
                self.agente.decay_epsilon()
                self.historial_epsilon.append(self.agente.epsilon)
            else:
                 # Añadir un valor nulo o 0 para mantener la longitud si se muestra
                 self.historial_epsilon.append(np.nan)


            # Loguear progreso cada N episodios
            if self.ultimo_episodio % 10 == 0:
                 logger.info(f"Episodio {self.ultimo_episodio}: Recompensa={recompensa_total_episodio:.2f}, Pérdida Media={perdida_media:.4f}"
                             f"{f', Epsilon={self.agente.epsilon:.3f}' if isinstance(self.agente, QLearningAgent) else ''}")

            # Actualizar gráficos y UI (quizás no en cada episodio si es muy rápido)
            if self.ultimo_episodio % 5 == 0: # Actualizar cada 5 episodios
                self.actualizar_graficos()
                self.actualizar_info_estado() # Muestra el estado final del episodio
                self.actualizar_vista_objetos() # Reflejar cambios en los objetos

            # Programar el siguiente ciclo si todavía estamos activos
            if self.entrenamiento_activo:
                self.root.after(5, self.ciclo_entrenamiento) # Pequeña pausa para responsividad

        except Exception as e:
            logger.exception("Error durante el ciclo de entrenamiento.")
            messagebox.showerror("Error de Entrenamiento", f"Ocurrió un error: {e}\nEl entrenamiento se detendrá.")
            self.detener_entrenamiento() # Detener en caso de error


    def detener_entrenamiento(self):
        if not self.entrenamiento_activo:
            return
        self.entrenamiento_activo = False
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)
        self._bloquear_controles_entorno(False) # Desbloquear edición
        logger.info("***** Entrenamiento detenido por el usuario *****")

    def _bloquear_controles_entorno(self, bloquear):
        """Activa o desactiva los controles de edición del entorno."""
        estado = tk.DISABLED if bloquear else tk.NORMAL
        # Botones de añadir/editar/eliminar objeto
        for btn in self.tree_objetos.master.master.winfo_children(): # Frame botones objetos
            if isinstance(btn, ttk.Button):
                btn.config(state=estado)
        # Botones importar/exportar entorno
        for btn in self.root.winfo_children()[0].winfo_children()[0].winfo_children()[-1].winfo_children(): # Frame guardar/cargar
             widget_text = btn.cget("text")
             if "Entorno" in widget_text:
                 btn.config(state=estado)
        # Quizás también los parámetros del agente si no queremos cambiarlos durante el entreno
        self.entry_lr.config(state=estado)
        self.entry_gamma.config(state=estado)
        self.entry_hidden_layers.config(state=estado)
        if self.agente_tipo.get() == "Q-Learning":
             self.entry_epsilon.config(state=estado)
             self.entry_epsilon_decay.config(state=estado)
        # Radiobuttons de tipo de agente
        for widget in self.root.winfo_children()[0].winfo_children()[0].winfo_children()[2].winfo_children()[0].winfo_children():
             if isinstance(widget, ttk.Radiobutton):
                  widget.config(state=estado)


    def actualizar_graficos(self):
        episodios = range(1, self.ultimo_episodio + 1)

        # Actualizar datos de las líneas
        self.line_reward.set_data(episodios, self.historial_recompensas)
        self.line_loss.set_data(episodios, self.historial_perdidas)
        if self.agente_tipo.get() == "Q-Learning":
             self.line_epsilon.set_data(episodios, [e for e in self.historial_epsilon if not np.isnan(e)])
             self.line_epsilon.set_visible(True)
        else:
             self.line_epsilon.set_visible(False)


        # Reescalar ejes
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        if self.agente_tipo.get() == "Q-Learning":
            # self.ax3.relim() # No necesario si Y es fijo
            # self.ax3.autoscale_view() # No necesario si Y es fijo
            pass # Y range is fixed
        self.ax3.set_xlim(0, self.ultimo_episodio + 1) # Asegurar que X se escala

        # Redibujar el canvas
        self.canvas.draw_idle() # Usar draw_idle para eficiencia


    def reiniciar_entorno_completo(self):
        """Reinicia todo: detiene entreno, recrea entorno, resetea agente, limpia gráficos."""
        logger.warning("Reiniciando entorno completo...")
        if self.entrenamiento_activo:
            self.detener_entrenamiento()

        # Recrear entorno desde los objetos actuales
        if self.objetos:
             try:
                  self.entorno = EntornoSimulado(self.objetos)
                  logger.info(f"Entorno recreado con {len(self.objetos)} objetos.")
             except ValueError as e:
                  logger.error(f"Error al recrear entorno durante reinicio: {e}")
                  messagebox.showerror("Error", f"No se pudo recrear el entorno: {e}")
                  self.entorno = None
        else:
             logger.warning("No hay objetos para recrear el entorno.")
             self.entorno = None

        # Resetear agente (eliminarlo para que se cree uno nuevo al entrenar)
        self.agente = None
        logger.info("Agente eliminado. Se creará uno nuevo al iniciar entrenamiento.")

        # Limpiar historial y gráficos
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.historial_epsilon = []
        self.ultimo_episodio = 0
        self.actualizar_graficos()

        # Actualizar UI
        self.actualizar_vista_objetos()
        self.actualizar_info_estado()

        logger.info("Reinicio completo finalizado.")

    def reiniciar_entorno(self):
        """Reinicia el estado del entorno y del agente, pero mantiene los objetos y la configuración."""
        logger.info("Reiniciando estado del entorno y agente...")
        if self.entrenamiento_activo:
            self.detener_entrenamiento()

        # Resetear estado del entorno
        if self.entorno:
            self.entorno.reset()

        # Resetear estado interno del agente (si existe)
        if self.agente:
             if isinstance(self.agente, QLearningAgent):
                  # Podríamos querer resetear epsilon aquí también
                  # self.agente.epsilon = float(self.entry_epsilon.get()) # Opcional
                  pass # QLearning no tiene mucho estado que resetear aparte de epsilon
             elif isinstance(self.agente, A2CAgent):
                  self.agente.reset() # Limpia buffers de trayectoria

        # Limpiar historial y gráficos
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.historial_epsilon = []
        self.ultimo_episodio = 0
        self.actualizar_graficos()

        # Actualizar UI
        self.actualizar_info_estado()
        self.actualizar_vista_objetos() # Para reflejar estado inicial si cambió

        logger.info("Reinicio de estado finalizado.")


    def ejecutar_accion_manual(self, accion):
        if not self.entorno:
            messagebox.showwarning("Advertencia", "El entorno no está inicializado.")
            return
        if self.entrenamiento_activo:
             messagebox.showwarning("Advertencia", "Detenga el entrenamiento para ejecutar acciones manuales.")
             return

        try:
             estado_ant = self.entorno.estado_actual
             cat_ant = self.objetos[estado_ant].obtener_categorias()[:] # Copia

             siguiente_estado, recompensa, done = self.entorno.ejecutar_accion(accion)

             estado_desp = self.entorno.estado_actual
             cat_desp = self.objetos[estado_desp].obtener_categorias()

             logger.info(f"Acción manual {accion} ejecutada. Recompensa: {recompensa:.2f}.")
             if estado_ant != estado_desp:
                  logger.info(f"  Cambio de objeto: {estado_ant} -> {estado_desp}")
             if cat_ant != cat_desp:
                  logger.info(f"  Cambio de categorías (Obj {estado_desp}): {cat_ant} -> {cat_desp}")


             # Actualizar UI
             self.actualizar_info_estado()
             self.actualizar_vista_objetos()

        except Exception as e:
            logger.exception(f"Error al ejecutar acción manual {accion}")
            messagebox.showerror("Error", f"Error ejecutando acción: {e}")

    def guardar_modelo(self):
        if not self.agente:
            messagebox.showinfo("Información", "No hay ningún agente entrenado para guardar.")
            return
        if self.entrenamiento_activo:
             messagebox.showwarning("Advertencia", "Detenga el entrenamiento antes de guardar el modelo.")
             return

        ruta = filedialog.asksaveasfilename(
            title="Guardar Modelo del Agente",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("Todos los archivos", "*.*")]
        )

        if ruta:
            try:
                self.agente.guardar_modelo(ruta)
                messagebox.showinfo("Éxito", f"Modelo guardado correctamente en:\n{ruta}")
            except Exception as e:
                logger.exception(f"Error al guardar el modelo en {ruta}")
                messagebox.showerror("Error", f"No se pudo guardar el modelo:\n{e}")

    def cargar_modelo(self):
        if self.entrenamiento_activo:
             messagebox.showwarning("Advertencia", "Detenga el entrenamiento antes de cargar un modelo.")
             return

        ruta = filedialog.askopenfilename(
            title="Cargar Modelo del Agente",
            filetypes=[("PyTorch Model", "*.pth"), ("Todos los archivos", "*.*")]
        )

        if ruta:
            try:
                # Determinar dimensiones actuales del entorno
                if not self.entorno:
                    messagebox.showerror("Error", "Cargue o cree un entorno antes de cargar un modelo.")
                    return
                current_state_dim = 1 # Asumiendo estado entero
                current_action_dim = self.entorno.num_acciones_disponibles

                # Cargar checkpoint para verificar compatibilidad antes de crear agente
                checkpoint = torch.load(ruta)
                agent_type = checkpoint.get('agent_type')
                model_state_dim = checkpoint.get('state_dim')
                model_action_dim = checkpoint.get('action_dim')
                model_hidden_layers = checkpoint.get('hidden_layers')

                if not agent_type or model_state_dim is None or model_action_dim is None or model_hidden_layers is None:
                     raise ValueError("Archivo de modelo incompleto o corrupto (faltan metadatos).")

                # Comprobar compatibilidad
                if model_state_dim != current_state_dim or model_action_dim != current_action_dim:
                     msg = (f"¡Incompatibilidad de Dimensiones!\n"
                            f"Modelo: Estado={model_state_dim}, Acciones={model_action_dim}\n"
                            f"Entorno: Estado={current_state_dim}, Acciones={current_action_dim}\n"
                            f"Cargar el modelo podría llevar a errores. ¿Continuar de todos modos?")
                     if not messagebox.askyesno("Advertencia de Compatibilidad", msg):
                          return
                     logger.warning(f"Cargando modelo con dimensiones incompatibles con el entorno actual.")


                # Crear el agente correcto y cargar
                if agent_type == "Q-Learning":
                     # Crear instancia vacía y luego cargar
                     self.agente = QLearningAgent(model_state_dim, model_action_dim, model_hidden_layers)
                     loaded_params = self.agente.cargar_modelo(ruta) # Carga estado, optimizador, etc.
                elif agent_type == "A2C":
                     self.agente = A2CAgent(model_state_dim, model_action_dim, model_hidden_layers)
                     loaded_params = self.agente.cargar_modelo(ruta)
                else:
                     raise ValueError(f"Tipo de agente desconocido en el archivo: {agent_type}")

                # Actualizar UI con parámetros cargados
                self.agente_tipo.set(agent_type)
                self.entry_lr.delete(0, tk.END)
                self.entry_lr.insert(0, str(loaded_params.get('learning_rate', '0.001')))
                self.entry_gamma.delete(0, tk.END)
                self.entry_gamma.insert(0, str(loaded_params.get('gamma', '0.99')))
                hidden_layers_str = ", ".join(map(str, model_hidden_layers))
                self.entry_hidden_layers.delete(0, tk.END)
                self.entry_hidden_layers.insert(0, hidden_layers_str)

                if agent_type == "Q-Learning":
                    self.entry_epsilon.delete(0, tk.END)
                    self.entry_epsilon.insert(0, f"{loaded_params.get('epsilon', 0.1):.3f}") # Mostrar epsilon cargado
                    self.entry_epsilon_decay.delete(0, tk.END)
                    self.entry_epsilon_decay.insert(0, str(loaded_params.get('epsilon_decay', '0.995')))
                self.toggle_epsilon_params() # Ajustar visibilidad

                # Reiniciar historial de entrenamiento
                self.historial_recompensas = []
                self.historial_perdidas = []
                self.historial_epsilon = []
                self.ultimo_episodio = 0
                self.actualizar_graficos()
                self.entorno.reset() # Resetear estado del entorno
                self.actualizar_info_estado()

                messagebox.showinfo("Éxito", f"Modelo {agent_type} cargado correctamente desde:\n{ruta}")

            except FileNotFoundError:
                 logger.error(f"Archivo de modelo no encontrado: {ruta}")
                 messagebox.showerror("Error", f"Archivo no encontrado:\n{ruta}")
            except TypeError as e:
                 logger.error(f"Error de tipo al cargar modelo: {e}")
                 messagebox.showerror("Error", f"Error de tipo de modelo:\n{e}")
            except ValueError as e:
                 logger.error(f"Error de valor al cargar modelo: {e}")
                 messagebox.showerror("Error", f"Error en el archivo del modelo:\n{e}")
            except Exception as e:
                logger.exception(f"Error inesperado al cargar el modelo desde {ruta}")
                messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")


    def exportar_entorno(self):
        if not self.objetos:
            messagebox.showinfo("Información", "No hay objetos en el entorno para exportar.")
            return
        if self.entrenamiento_activo:
            messagebox.showwarning("Advertencia", "Detenga el entrenamiento antes de exportar el entorno.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Exportar Configuración del Entorno",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Todos los archivos", "*.*")]
        )

        if ruta:
            try:
                datos_objetos = [obj.to_dict() for obj in self.objetos]
                # Podríamos añadir metadatos del entorno si fuera necesario
                entorno_data = {
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "objetos": datos_objetos
                    # Añadir aquí otros parámetros del entorno si los hubiera
                }
                with open(ruta, 'w', encoding='utf-8') as f:
                    json.dump(entorno_data, f, indent=4)
                logger.info(f"Entorno exportado a {ruta} con {len(datos_objetos)} objetos.")
                messagebox.showinfo("Éxito", f"Entorno exportado correctamente a:\n{ruta}")
            except Exception as e:
                logger.exception(f"Error al exportar el entorno a {ruta}")
                messagebox.showerror("Error", f"No se pudo exportar el entorno:\n{e}")

    def importar_entorno(self):
        if self.entrenamiento_activo:
             messagebox.showwarning("Advertencia", "Detenga el entrenamiento antes de importar un entorno.")
             return

        ruta = filedialog.askopenfilename(
            title="Importar Configuración del Entorno",
            filetypes=[("JSON files", "*.json"), ("Todos los archivos", "*.*")]
        )

        if ruta:
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    entorno_data = json.load(f)

                if "objetos" not in entorno_data:
                     raise ValueError("El archivo JSON no contiene la clave 'objetos'.")

                datos_objetos = entorno_data["objetos"]
                if not datos_objetos:
                    raise ValueError("La lista de objetos en el archivo está vacía.")

                # Validar consistencia de estructura (num_cat, bits_cat) entre objetos importados
                if len(datos_objetos) > 1:
                     first_obj_data = datos_objetos[0]
                     num_cat_ref = first_obj_data.get("num_categorias")
                     bits_cat_ref = first_obj_data.get("bits_por_categoria")
                     for i, obj_data in enumerate(datos_objetos[1:], 1):
                          if obj_data.get("num_categorias") != num_cat_ref or \
                             obj_data.get("bits_por_categoria") != bits_cat_ref:
                              raise ValueError(f"Inconsistencia en la estructura de los objetos importados (objeto índice {i}). Todos deben tener el mismo número de categorías y bits.")

                # Crear nuevos objetos desde los datos
                nuevos_objetos = [ObjetoBinario.from_dict(d) for d in datos_objetos]

                # Reemplazar objetos actuales y reiniciar todo
                self.objetos = nuevos_objetos
                logger.info(f"Entorno importado desde {ruta}. {len(self.objetos)} objetos cargados.")
                self.reiniciar_entorno_completo() # Reinicia entorno, agente, gráficos, UI

                messagebox.showinfo("Éxito", f"Entorno importado correctamente desde:\n{ruta}")

            except FileNotFoundError:
                 logger.error(f"Archivo de entorno no encontrado: {ruta}")
                 messagebox.showerror("Error", f"Archivo no encontrado:\n{ruta}")
            except json.JSONDecodeError as e:
                 logger.error(f"Error al decodificar JSON desde {ruta}: {e}")
                 messagebox.showerror("Error", f"Archivo JSON inválido:\n{e}")
            except ValueError as e:
                 logger.error(f"Error en los datos del entorno importado: {e}")
                 messagebox.showerror("Error", f"Error en los datos del archivo:\n{e}")
            except Exception as e:
                logger.exception(f"Error inesperado al importar el entorno desde {ruta}")
                messagebox.showerror("Error", f"No se pudo importarl el entorno:\n{e}")

    def on_closing(self):
        # Acciones a realizar al cerrar la ventana
        logger.info("Cerrando aplicación...")
        if self.entrenamiento_activo:
            logger.warning("El entrenamiento estaba activo. Deteniendo...")
            self.detener_entrenamiento() # Asegurarse de que el flag se ponga a False

        # Podríamos guardar el estado automáticamente aquí si quisiéramos

        # Cerrar handlers de logging para liberar archivos
        logger.info("Cerrando handlers de logging...")
        for handler in logger.handlers[:]:
             try:
                 handler.close()
                 logger.removeHandler(handler)
             except Exception as e:
                  print(f"Error cerrando handler {handler}: {e}") # Usar print porque el logger puede estar ya cerrado

        self.root.destroy()
        print("Aplicación cerrada.")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = QuantumAgentApp(root)
        root.mainloop()
    except Exception as e:
        logger.critical(f"Error fatal en la aplicación: {e}", exc_info=True)
        # Intentar mostrar un mensaje de error final si Tkinter aún funciona
        try:
             messagebox.showerror("Error Fatal", f"La aplicación encontró un error crítico y se cerrará:\n{e}")
        except:
             pass # Si Tkinter falló, no se puede mostrar messagebox