import numpy as np

# Prima idea di rete neurale che tratta dati sintetici
np.random.seed(0)

# Generazione inputs tramite spirale
def create_data(points, classes): 

    X = np.zeros((points * classes, 2))
    y = np.zeros(points  * classes, dtype = 'uint8')

    for class_number in range(classes): 
        ix     =  range(points * class_number, points * (class_number + 1))
        r      =  np.linspace(0.0, 1, points)
        t      =  np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix]  =  np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix]  =  class_number

    return X, y


# Definizione dei layer e activations
class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs): 
        # Copia di input
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases # si muove in avanti

    def backward(self, dvalues): 
        # Calcolo del gradiente sui parametri
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        # Calcolo del gradiente sugli inputs
        self.dinputs  = np.dot(dvalues, self.weights.T)


class Activation_ReLU: 

    def forward(self, inputs): 
        #Copia degli input
        self.inputs = inputs
        self.output = np.maximum(0, inputs) # azzera i negativi

    def backward(self, dvalues): 
        # Prepariamo una copia
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 #azzero i negativi


class Activation_Softmax: 

    def forward(self, inputs): 
        # Copia dei valori in input
        self.inputs   = inputs

        exp_values    = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True)) # input - il massimo input per riga, per evitare overflow
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output   = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) #array inizializzato

        # Enumerazione degli output e dei gradienti
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): 
            # Collasso l'array
            single_output       =  single_output.reshape(-1, 1)
            # Calcolo la matrice Jacobiana dell'output
            jacobian_matrix     =  np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] =  np.dot(jacobian_matrix, single_dvalues)

# # # # # #
# Ho scelto di utilizzare Adam come ottimizzatore
# per l'autoencoder. Credo sia la scelta più adatta 
# grazie alla sua versatilità.
# Attualmente mi concentrerò su questo tipo di ottimizzatore.
class Optimizer_Adam: 

    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate          =  learning_rate
        self.current_learning_rate  =  learning_rate
        self.decay                  =  decay
        self.iterations             =  0
        self.epsilon                =  epsilon
        self.beta_1                 =  beta_1
        self.beta_2                 =  beta_2
        
    # Gestione del decadimento del learning_rate
    # Quando l'algoritmo si avvicina al minimo il passo diminuisce
    def pre_update_params(self): 
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer): 
        # Creazione memoria per i gradienti precedenti
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)
    
        # Aggiornamento momentums con i gradienti correnti 
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums   + (1 - self.beta_1) * layer.dbiases
        # -> m_t = B_1 * m_t-1 + (1 - B_1) * g_t
        
        # Correzione momentums 
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected   = layer.bias_momentums   / (1 - self.beta_1 ** (self.iterations + 1))
        # -> <m_t = m_t / (1 - B_1 ^ t)

        # Aggiorniamo la memoria con i quadrati dei gradienti
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache   + (1 - self.beta_2) * layer.dbiases  ** 2
        # -> v_t = B_2 * v_t-1 + (1 - B_2) * g_t ^ 2

        # Correzione bias del momento 2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache     / (1 - self.beta_2 ** (self.iterations + 1))
        # -> <v_t = v_t / (1 - B_2 ^ t)

        # Aggiornamento pesi e bias
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases  += -self.current_learning_rate * bias_momentums_corrected   / (np.sqrt(bias_cache_corrected)   + self.epsilon)
        # -> 0_t+1 = 0_t - [n / (sqrt(<v_t) + epsilon)] * <m_t
        # Formula di Adam: evita le oscillazioni e migliora la convergenza

    def post_update_params(self): 
        self.iterations += 1



# Calcolo Loss 
class Loss: 

    def calculate(self, output, y): 
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true): 
        samples         =  len(y_pred)
        y_pred_clipped  =  np.clip(y_pred, 1e-7, 1-1e-7) # per evitare errori come log(1) o log(0)

        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        
        negative_log_likehoods = -np.log(correct_confidences)
        return negative_log_likehoods
    
    def backward(self, dvalues, y_true): 
        samples = len(dvalues)
        # Numero di classi in ogni campione
        labels  = len(dvalues[0]) # viene utilizzato il primo campione per contarli

        # se i campioni sono sparsi, li trasformiamo in un array one-hot
        if len(y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true]

        # Calcolo del gradiente e normalizzazione
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


# Combinazione di Softmax e cross-entropy loss per un backward piu' veloce
class Activation_Softmax_Loss_CategoricalCrossentropy(): 

    def __init__(self): 
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Output layer di attivazione
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true): 
        samples = len(dvalues)

        # Trasformo le matrici one-hot in arrays
        if len(y_true.shape) == 2: 
            y_true = np.argmax(y_true, axis = 1)

        # Copio i valori
        self.dinputs = dvalues.copy()

        # Calcolo e normalizzo il gradiente
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# # #
# Flusso 
# # # # # # # #

# Dataset a spirale
X, y = create_data(points = 100, classes = 3)

# Creazione del primo layer denso con 2 input e 64 output
layer1_in  = Layer_Dense(2, 64)
layer1_out = Activation_ReLU()

# Creazione del secondo layer denso con 64 input e 3 output
layer2_in  = Layer_Dense(64, 3)
loss_layer2_out = Activation_Softmax_Loss_CategoricalCrossentropy()

# Definizione dell'ottimizzatore
optimizer = Optimizer_Adam(learning_rate = 0.05, decay = 5e-7)

# Train
for epoch in range(50001): 

    layer1_in.forward(X)
    layer1_out.forward(layer1_in.output)

    layer2_in.forward(layer1_out.output) ##
    loss = loss_layer2_out.forward(layer2_in.output, y)

    # Calcolo precisione dall'output di Softmax 
    predictions = np.argmax(loss_layer2_out.output, axis = 1)
    if len(y.shape) == 2: 
        y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100: 
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

    # Backward
    loss_layer2_out.backward(loss_layer2_out.output, y)
    layer2_in.backward(loss_layer2_out.dinputs)
    layer1_out.backward(layer2_in.dinputs)
    layer1_in.backward(layer1_out.dinputs)

    # Aggiornamento pesi e bias
    optimizer.pre_update_params()
    optimizer.update_params(layer1_in)
    optimizer.update_params(layer2_in)
    optimizer.post_update_params()
