import numpy as np
import matplotlib.pyplot as plt

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

# Generazione dati sinusoidali
def sine_data(samples=1000):
    X = np.linspace(0, 4 * np.pi, samples)  
    y = np.sin(X) 
    X = X.reshape(-1, 1)  
    y = y.reshape(-1, 1)  

    return X, y




# Definizione dei layer e activations
class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0, weight_regularizer_l2 = 0, bias_regularizer_l1 = 0, bias_regularizer_l2 = 0): 
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

        # Definizione dei regolarizzatori
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1   = bias_regularizer_l1
        self.bias_regularizer_l2   = bias_regularizer_l2

    def forward(self, inputs): 
        # Copia di input
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases # si muove in avanti

    def backward(self, dvalues): 
        # Calcolo del gradiente sui parametri
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)

        # Gradienti nei regolarizzatori
        # L1 sui pesi
        if self.weight_regularizer_l1 > 0: 
            dL1                   = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights        += self.weight_regularizer_l1 * dL1
        
        #L2 sui pesi
        if self.weight_regularizer_l2 > 0: 
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        #L1 sui bias
        if self.bias_regularizer_l1 > 0: 
            dL1                 = np.ones_like(self.biases)
            dL1[self.biass < 0] = -1
            self.dbiases       += self.bias_regularizer_l1 * dL1
        
        #L2 sui bias
        if self.bias_regularizer_l2 > 0: 
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Calcolo del gradiente sugli inputs
        self.dinputs  = np.dot(dvalues, self.weights.T)


class Layer_Dropout: 

    def __init__(self, rate): 
        # Salvo il success rate per il dropout
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs      = inputs # Copia dei valori in input
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
        self.output      = inputs * self.binary_mask

    def backward(self, dvalues): 
        self.dinputs = dvalues * self.binary_mask


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


class Activation_Linear: 

    def forward(self, inputs): 
        self.inputs = inputs # Copia valore
        self.output = inputs

    def backward(self, dvalues):
        # Derivata e' 1 che moltiplicato per dvalue rimane invariato (chain rule)
        self.dinputs = dvalues.copy()

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

    # Regolarizzatore per il calcolo su loss
    def regularization_loss(self, layer): 
        # default
        regularization_loss = 0

        # L1 - pesi
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        
        # L2 - pesi
        if layer.weight_regularizer_l2 > 0: 
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # L1 - bias
        if layer.bias_regularizer_l2 > 0: 
            regularization_loss += layer.bias_regularizer_l1   * np.sum(np.abs(layer.biases))

        # L2 - bias
        if layer.bias_regularizer_l2 > 0: 
            regularization_loss += layer.bias_regularizer_l2   * np.sum(layer.biases * layer.biases)

        return regularization_loss

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

# Adotto Mean Squared Error loss (piu' adatto ad un Autoencoder)
class Loss_MeanSquaredError(Loss): # con L2

    def forward(self, y_pred, y_true): 
        sample_losses = np.mean((y_true - y_pred) ** 2, axis = -1)
        return sample_losses
    
    def backward(self, dvalues, y_true): 
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Calcolo del gradiente e normalizzazione
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

# # #
# Flusso 
# # # # # # # #

# Dataset sinusoidale
X, y = sine_data()

# Creazione del primo layer denso con 1 input e 64 output
layer1_in  = Layer_Dense(1, 64)
layer1_out = Activation_ReLU()

# Creazione del layer di Dropout
# dropout_layer = Layer_Dropout(0.1)

# Creazione del secondo layer denso con 64 input e 64 output
layer2_in  = Layer_Dense(64, 64)
layer2_out = Activation_ReLU()

# Creazione del terzo layer denso con 64 input e 1 output
layer3_in  = Layer_Dense(64, 1)
layer3_out = Activation_Linear()

loss_function = Loss_MeanSquaredError()

# Definizione dell'ottimizzatore
optimizer = Optimizer_Adam(learning_rate = 0.005, decay = 1e-3)

# Definizione del parametro di precisione (simulato)
accuracy_precision = np.std(y) / 250

# Train
for epoch in range(10001): 

    layer1_in.forward(X)
    layer1_out.forward(layer1_in.output)
    layer2_in.forward(layer1_out.output)
    layer2_out.forward(layer2_in.output)
    layer3_in.forward(layer2_out.output)
    layer3_out.forward(layer3_in.output)

    data_loss = loss_function.calculate(layer3_out.output, y)
    regularization_loss = loss_function.regularization_loss(layer1_in) + loss_function.regularization_loss(layer2_in) + loss_function.regularization_loss(layer3_in)
    loss = data_loss + regularization_loss

    # Calcolo precisione utilizzando <x = | y_pred - y_true | 
    predictions = layer3_out.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100: 
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, data_loss: {regularization_loss:.3f}, reg_los: {regularization_loss:.3f}, lr: {optimizer.current_learning_rate}')

    # Backward
    loss_function.backward(layer3_out.output, y)
    layer3_out.backward(loss_function.dinputs)
    layer3_in.backward(layer3_out.dinputs)
    layer2_out.backward(layer3_in.dinputs)
    layer2_in.backward(layer2_out.dinputs)
    layer1_out.backward(layer2_in.dinputs)
    layer1_in.backward(layer1_out.dinputs)

    # Aggiornamento pesi e bias
    optimizer.pre_update_params()
    optimizer.update_params(layer1_in)
    optimizer.update_params(layer2_in)
    optimizer.update_params(layer3_in)
    optimizer.post_update_params()

# # # #
# Validazione del modello

X_test, y_test = sine_data()

layer1_in.forward(X_test)
layer1_out.forward(layer1_in.output)
layer2_in.forward(layer1_out.output)
layer2_out.forward(layer2_in.output)
layer3_in.forward(layer2_out.output)
layer3_out.forward(layer3_in.output)

plt.plot(X_test, y_test)
plt.plot(X_test, layer3_out.output)
plt.show()

''' 
data_loss = loss_function.calculate(layer3_out.output, y)

# Calcolo precisione
predictions = layer3_out.output
accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
'''