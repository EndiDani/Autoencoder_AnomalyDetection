import numpy as np

# Prima idea di rete neurale che tratta dati sintetici
np.random.seed(0)

# Generazione inputs tramite spirale
def create_data(points, classes): 

    X = np.zeros((points * classes, 2))
    y = np.zeros(points  * classes, dtype = 'uint8')

    for class_number in range(classes): 
        ix = range(points * class_number, points * (class_number + 1))
        r  = np.linspace(0.0, 1, points)
        t  = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y


# Definizione dei layer e activations
class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs): 
        # Copia di input
        self.inputs = inputs 
        self.output  = np.dot(inputs, self.weights) + self.biases # si muove in avanti

    def backward(self, dvalues): 
        # Calcolo del gradiente sui parametri
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        # Calcolo del gradiente sugli inputs
        self.dinputs = np.dot(dvalues, self.weights.T)


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
        self.inputs = inputs

        exp_values    = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True)) # input - il massimo input per riga, per evitare overflow
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output   = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) #array inizializzato

        # Enumerazione degli output e dei gradienti
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): 
            # Collasso l'array
            single_output = single_output.reshape(-1, 1)
            # Calcolo la matrice Jacobiana dell'output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Calcolo Loss (ripassa questa parte)
class Loss: 

    def calculate(self, output, y): 
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true): 
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # per evitare errori come log(1) o log(0)

        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        
        negative_log_likehoods = -np.log(correct_confidences)
        return negative_log_likehoods
    
    def backward(self, dvalues, y_true): 
        samples = len(dvalues)
        # Numero di classi in ogni campione
        labels = len(dvalues[0]) # viene utilizzato il primo campione per contarli

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
        self.loss = Loss_CategoricalCrossentropy()

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

X, y = create_data(points = 100, classes = 3)

layer1_in  = Layer_Dense(2, 3)
layer1_out = Activation_ReLU()

layer2_in  = Layer_Dense(3, 3)
loss_layer2_out = Activation_Softmax_Loss_CategoricalCrossentropy()

layer1_in.forward(X)
layer1_out.forward(layer1_in.output)

layer2_in.forward(layer1_out.output)

loss = loss_layer2_out.forward(layer2_in.output, y)

# Print dei primi 5 punti e loss calcolata
print(loss_layer2_out.output[:5])
print("Loss: ", loss)

# Precisione ottenuta dall'output della Softmax 
predictions = np.argmax(loss_layer2_out.output, axis = 1)
if len(y.shape) == 2: 
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions == y)
print("Accuracy: ", accuracy)

# Backward pass
loss_layer2_out.backward(loss_layer2_out.output, y)
layer2_in.backward(loss_layer2_out.dinputs)
layer1_out.backward(layer2_in.dinputs)
layer1_in.backward(layer1_out.dinputs)

# Stampo gradienti
print(layer1_in.dweights)
print(layer1_in.dbiases)
print(layer2_in.dweights)
print(layer2_in.dbiases)


'''
class Backpropagation: 

    def calculate_gradients(self, y_pred, y_true, layer1_out, layer1_in, layer2_in):
        # Calcolo del gradiente per il secondo layer
        y_true_one_hot = np.eye(3)[y_true]  
        gradient_lastLayer  = y_pred.output - y_true_one_hot

        # Gradiente per i pesi e bias del secondo layer
        layer2_in.weights_gradient = np.dot(layer1_out.output.T, gradient_lastLayer)
        layer2_in.biases_gradient  = np.sum(gradient_lastLayer, axis = 0, keepdims = True)

        # Gradiente per il primo layer (passando attraverso la ReLU)
        gradient_firstLayer = np.dot(gradient_lastLayer, layer2_in.weights.T) * (layer1_in.output > 0).astype(float)
                                                                                 # maybe layer1_out                       
        # Gradiente per i pesi e bias del primo layer
        layer1_in.weights_gradient = np.dot(layer1_out.output.T, gradient_firstLayer) # forse X.T?
        layer1_in.biases_gradient  = np.sum(gradient_firstLayer, axis = 0, keepdims = True)

    def apply_gradients(self, layer, learning_rate): 
        # Aggiornamento pesi e bias utilizzando il gradiente
        layer.weights -= learning_rate * layer.weights_gradient
        layer.biases  -= learning_rate * layer.biases_gradient


# # #
# Flusso 
# # # # # # # #

# Generazione dei dati
X, y = create_data(points = 100, classes = 3)

# Inizializzazione dei layer
layer1_in  = Layer_Dense(2, 3)
layer1_out = Activation_ReLU()

layer2_in  = Layer_Dense(3, 3)
layer2_out = Activation_Softmax()

# Funzione di perdita
loss_function = Loss_CategoricalCrossentropy()

# Ciclo di addestramento
epochs = 100
learning_rate = 0.1
backprop = Backpropagation()

for epoch in range(epochs): # Per ogni epoca il modello vedrà i dati una volta
    
    layer1_in.forward(X)
    layer1_out.forward(layer1_in.output)
    layer2_in.forward(layer1_out.output)
    layer2_out.forward(layer2_in.output)

    loss = loss_function.calculate(layer2_out.output, y)

    backprop.calculate_gradients(layer2_out, y, layer1_out, layer1_in, layer2_in)
    
    backprop.apply_gradients(layer1_in, learning_rate)
    backprop.apply_gradients(layer2_in, learning_rate)


    if epoch % 10 == 0: 
        print(f"Epoch {epoch} - Loss: {loss}")

'''
