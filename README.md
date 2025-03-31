
**Explicación Funcional

Este script crea una aplicación de escritorio con una interfaz gráfica (GUI) que simula un entorno simple y permite entrenar a un "agente" inteligente para interactuar con él usando técnicas de Aprendizaje por Refuerzo (Reinforcement Learning - RL).

1.  **Entorno Simulado:**
    *   El entorno consiste en una serie de "Objetos Binarios".
    *   Cada objeto tiene un nombre y varias "categorías".
    *   El valor de cada categoría se representa internamente como una cadena de bits (ceros y unos), pero se muestra al usuario como un número decimal.
    *   El agente sólo puede "ver" o interactuar con un objeto a la vez.

2.  **Agente Inteligente:**
    *   El objetivo del agente es aprender a realizar acciones dentro de este entorno para maximizar una "recompensa" acumulada.
    *   Puede realizar acciones como:
        *   **Moverse:** Cambiar el objeto en el que está enfocado (izquierda/derecha).
        *   **Ajustar:** Incrementar o decrementar el valor numérico de una categoría específica del objeto actual.

3.  **Aprendizaje por Refuerzo (RL):**
    *   El usuario puede elegir entre dos algoritmos de RL:
        *   **Q-Learning:** Un algoritmo clásico basado en valores que aprende a estimar la calidad (Q-value) de realizar cada acción en cada estado posible.
        *   **Actor-Crítico (A2C):** Un algoritmo más avanzado que tiene dos partes: un "Actor" que decide qué acción tomar y un "Crítico" que evalúa qué tan bueno es el estado actual.
    *   El agente aprende a través de prueba y error. Realiza acciones, recibe recompensas (o penalizaciones pequeñas) del entorno, y ajusta su "cerebro" (la red neuronal) para tomar mejores decisiones en el futuro.

4.  **Interfaz Gráfica (GUI):**
    *   Permite al usuario:
        *   **Configurar el Entorno:** Añadir, editar (cambiar nombre y valores de categorías) y eliminar los objetos binarios.
        *   **Configurar el Agente:** Elegir el algoritmo (Q-Learning/A2C), ajustar parámetros clave (tasa de aprendizaje, factor de descuento gamma, parámetros de exploración epsilon para Q-Learning) y definir la estructura de la red neuronal (número de capas y neuronas).
        *   **Entrenar:** Iniciar y detener el proceso de entrenamiento del agente.
        *   **Visualizar:** Ver gráficos en tiempo real que muestran el progreso del entrenamiento (recompensa por episodio, pérdida de la red neuronal, nivel de exploración epsilon).
        *   **Control Manual:** Ejecutar acciones específicas manualmente para probar el entorno o el agente entrenado.
        *   **Guardar/Cargar:** Salvar el estado del agente entrenado (su red neuronal y parámetros) a un archivo y cargarlo posteriormente. También permite exportar e importar la configuración del entorno (la lista de objetos).
        *   **Ver Logs:** Consultar un registro de eventos importantes que ocurren durante la ejecución y el entrenamiento.

**En resumen:** Es una herramienta visual para experimentar con algoritmos básicos de RL en un entorno configurable simple, permitiendo observar cómo un agente aprende a manipular valores numéricos dentro de objetos definidos.

**Explicación Técnica Breve**

1.  **Interfaz (Tkinter):** La GUI se construye usando la biblioteca estándar de Python `tkinter`. Se utilizan widgets como `ttk.Frame`, `ttk.Label`, `ttk.Button`, `ttk.Entry`, `ttk.Treeview` (para la lista de objetos), `scrolledtext.ScrolledText` (para los logs) y `ttk.Notebook` (para las pestañas). `matplotlib.backends.backend_tkagg` se usa para incrustar los gráficos de `matplotlib` dentro de la ventana de Tkinter.
2.  **Entorno (`EntornoSimulado`, `ObjetoBinario`):**
    *   `ObjetoBinario`: Almacena el estado de un objeto (nombre, número de categorías, bits por categoría, y la lista de valores de categoría como strings binarios). Proporciona métodos para actualizar categorías y obtener la representación binaria completa.
    *   `EntornoSimulado`: Gestiona la colección de `ObjetoBinario`. Mantiene el índice del objeto actual (`estado_actual`). Define la función `ejecutar_accion` que traduce un número de acción (ej: 0 para derecha, 2 para incrementar categoría 0) en cambios en el estado del entorno (cambiar `estado_actual` o llamar a `actualizar_categoria` del objeto) y devuelve el nuevo estado, una recompensa numérica y un indicador de finalización (siempre `False` aquí). El `obtener_estado` devuelve la representación numérica (entero) del string binario del objeto actual.
3.  **Redes Neuronales (PyTorch):**
    *   `QNetwork`: Un modelo de red neuronal simple (Perceptrón Multicapa - MLP) implementado con `torch.nn.Module`. Recibe el estado (el entero del objeto binario, tratado como un tensor de flotantes `[batch_size, 1]`) y produce un Q-value para cada acción posible. Usa capas `nn.Linear`, activación `nn.ReLU` y `nn.Dropout`.
    *   `ActorCritic`: Otro MLP que tiene dos "cabezas" de salida: una para el Actor (produce logits, que determinan la probabilidad de cada acción) y otra para el Crítico (produce una estimación del valor del estado actual).
4.  **Agentes de RL (PyTorch):**
    *   `QLearningAgent`: Implementa Q-Learning.
        *   **Selección de Acción:** Usa epsilon-greedy (explora aleatoriamente con probabilidad epsilon, o elige la acción con el Q-value máximo estimado por `q_network`).
        *   **Memoria:** Usa una `collections.deque` como búfer de repetición para almacenar tuplas `(estado, acción, recompensa, siguiente_estado, done)`.
        *   **Aprendizaje:** Muestrea lotes (batches) de la memoria. Calcula los Q-values objetivo usando la red `target_q_network` (una copia periódicamente actualizada de `q_network` para estabilidad) y la ecuación de Bellman (`target = reward + gamma * max_Q_target(next_state)`). Calcula la pérdida (MSE) entre los Q-values actuales (`q_network(state)`) y los objetivos, y actualiza `q_network` usando el optimizador Adam. Realiza una "actualización suave" (soft update) de `target_q_network`.
    *   `A2CAgent` (Advantage Actor-Critic):
        *   **Selección de Acción:** Obtiene logits y valor del estado desde `actor_critic`. Usa `torch.distributions.Categorical` para muestrear una acción basada en los logits del Actor. Almacena el logaritmo de la probabilidad de la acción elegida (`log_prob`) y el valor del estado (`value`).
        *   **Almacenamiento:** Guarda recompensas, indicadores `done`, `log_probs` y `values` para una secuencia de pasos.
        *   **Aprendizaje:** Al final de una secuencia (episodio), calcula los "retornos" (recompensas futuras descontadas) y las "ventajas" (`Advantage = Return - Value`). La pérdida del Actor se calcula para aumentar la probabilidad de acciones con ventajas positivas. La pérdida del Crítico se calcula como el MSE entre los valores predichos y los retornos calculados. Ambas pérdidas se combinan y se usa Adam para actualizar la red `actor_critic`.
5.  **Integración:** La clase `QuantumAgentApp` orquesta todo: maneja eventos de la GUI, instancia el entorno y el agente, ejecuta el ciclo de entrenamiento (usando `root.after` para no bloquear la GUI), actualiza los gráficos y logs, y gestiona el guardado/cargado.
6.  **Manejo de Datos:** `json` se usa para serializar/deserializar la configuración del entorno (objetos). `torch.save`/`load` se usan para guardar/cargar el estado completo del modelo PyTorch (pesos de la red, estado del optimizador, hiperparámetros como epsilon).
7.  **Logging:** El módulo `logging` se configura para enviar mensajes a un archivo (`quantum_agent.log`) y a la pestaña "Log" de la GUI a través de un `LogTextHandler` personalizado.