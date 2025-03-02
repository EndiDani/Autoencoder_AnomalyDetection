import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import copy

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

    def forward(self, inputs, training): 
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

    def forward(self, inputs, training):
        self.inputs      = inputs # Copia dei valori in input

        if not training: 
            self.output  = inputs.copy()
            return 
        
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
        self.output      = inputs * self.binary_mask

    def backward(self, dvalues): 
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:

    def forward(self, inputs, training): 
        self.output = inputs


class Activation_ReLU: 

    def forward(self, inputs, training): 
        #Copia degli input
        self.inputs = inputs
        self.output = np.maximum(0, inputs) # azzera i negativi

    def backward(self, dvalues): 
        # Prepariamo una copia
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 #azzero i negativi

    def predictions(self, outputs): 
        return outputs


class Activation_Softmax: 

    def forward(self, inputs, training): 
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

    def predictions(self, outputs): 
        return np.argmax(outputs, axis = 1)

class Activation_Linear: 

    def forward(self, inputs, training): 
        self.inputs = inputs # Copia valore
        self.output = inputs

    def backward(self, dvalues):
        # Derivata e' 1 che moltiplicato per dvalue rimane invariato (chain rule)
        self.dinputs = dvalues.copy()

    def predictions(self, outputs): 
        return (outputs > 0.5) * 1

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
    def regularization_loss(self): 
        # default
        regularization_loss = 0

        for layer in self.trainable_layers: 

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

    def remember_trainable_layers(self, trainable_layers): 
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization = False): 
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)

        self.accumulated_sum   += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization: 
            return data_loss
        
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization = False): 
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization: 
            return data_loss
        
        return data_loss, self.regularization_loss()

    def new_pass(self): 
        self.accumulated_sum   = 0
        self.accumulated_count = 0
    
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


class Accuracy: 

    def calculate(self, predictions, y):
        comparison = self.compare(predictions, y)
        accuracy   = np.mean(comparison)
        return accuracy


class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit = False):

        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision



class Model: 

    def __init__(self): 
        self.layers = []
        self.softmax_classifier_output = None
    
    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer, accuracy): 
        self.loss      = loss
        self.optimizer = optimizer
        self.accuracy  = accuracy

    def finalize(self): 
        self.input_layer      = Layer_Input()
        layer_count           = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count): 

            # Se e' il primo layer il layer precedente era l'inputLayer
            if i == 0: 
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # Per tutti i layer eccetto il primo e l'ultimo
            elif i < layer_count - 1: 
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # L'ultimo layer contiene il loss con oltre a stampare l'output
            else: 
                self.layers[i].prev           = self.layers[i - 1]
                self.layers[i].next          = self.loss
                self.output_layer_activation = self.layers[i]

            # Se il layer ha un attributo chiamato "weights" (riferito al peso)
            # e' definito allenabile
            if hasattr(self.layers[i], 'weights'): 
                self.trainable_layers.append(self.layers[i])
        
        # Aggiornamento loss con i layer allenabili
        self.loss.remember_trainable_layers(self.trainable_layers)

    def train(self, X, y, *, epochs = 1, print_every = 1, validation_data = None): 
        
        self.accuracy.init(y)

        for epoch in range(1, epochs + 1):
            # Forward pass
            output = self.forward(X, training = True)

            # Calcolo loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization = True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy    = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            # Ottimizazzione
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers: 
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every: 
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, data_loss: {regularization_loss:.3f}, reg_los: {regularization_loss:.3f}, lr: {self.optimizer.current_learning_rate}')

        # Se c'e' la validazione dei dati
        if validation_data is not None: 
            # Leggibilita'
            X_val, y_val  =  validation_data
            output        =  self.forward(X_val, training=False)
            loss          =  self.loss.calculate(output, y_val)
            predictions    =  self.output_layer_activation.predictions(output)
            accuracy      =  self.accuracy.calculate(predictions, y_val)

            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

    def forward(self, X, training): 
        # Chiamata al forward pass del layer di input
        self.input_layer.forward(X, training)

        # Chiamata al forward pass per ogni layer nella catena
        # Ogni output viene passato come parametro al layer successivo
        for layer in self.layers: 
            layer.forward(layer.prev.output, training)

        return layer.output #ultimo layer

    def backward(self, output, y): 
        # Salto i controlli per softmax e attivazioni
        # che non mi serviranno per il mio autoencoder
        self.loss.backward(output, y)

        for layer in reversed(self.layers): 
            layer.backward(layer.next.dinputs)

    # Sulla evaluation mi concentrero' solo sul loss
    def evaluate(self, X_val, y_val, *, batch_size = None):
        validation_steps = 1

        # Calcolo numero di step
        if batch_size is not None: 
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val): 
                validation_steps += 1
        
        self.loss.new_pass()

        for step in range(validation_steps): 
            # Se batch_size non e' settato utilizzo uno step e tutto il dataset
            if batch_size is None: 
                batch_X = X_val
                batch_y = y_val
            
            # Altrimenti divido il batch
            else: 
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]
            
            output = self.forward(batch_X, training = False)
            # Calcolo della loss
            self.loss.calculate(output, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        print(f'validation, loss: {validation_loss:.3f}')
    
    def predict(self, X, *, batch_size = None): 
        prediction_steps = 1

        if batch_size is not None: 
            prediction_steps = len(X) // batch_size
            
            if prediction_steps * batch_size < len(X): 
                prediction_steps += 1

        # Output del modello
        output = []

        for step in range(prediction_steps): 
            if batch_size is None: 
                batch_X = X
            else: 
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            batch_output = self.forward(batch_X, training = False)
            # Append nella lista delle predizioni
            output.append(batch_output)
        
        # Si stackano le previsioni e si ritornano
        return np.vstack(output)

    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    
    def set_parameters(self, parameters): 
        
        for parameter_set, layer in zip(parameters, self.trainable_layers): 
            layer.set_parameters(*parameter_set)
    
    # Salva i parametri in un file
    def save_parameters(self, path): 

        with open(path, "wb") as f: 
            pickle.dump(self.get_parameters(), f)
    
    # Carica i parametri
    def load_parameters(self, path): 

        with open(path, "rb") as f: 
            self.set_parameters(pickle.loads(f))

    # Salva il modello
    def save(self, path): 
        
        model = copy.deepcopy(self)

        # Reset dei valori accumulati in loss
        model.loss.new_pass()

        # Pulizia dell'input layer e dei gradienti 
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # Per ogni layer rimuove gli input, output e dinputs
        for layer in model.layers: 
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']: 
                layer.__dict__.pop(property, None)
        
        # Salvataggio del modello in un file
        with open(path, 'wb') as f: 
            pickle.dump(model, f)
        
    # Ritorna il modello salvato
    @staticmethod
    def load(path): 
        
        with open(path, 'rb') as f: 
            model = pickle.load(f)
        return model
    



