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
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs): 
        self.output  = np.dot(inputs, self.weights) + self.biases # si muove in avanti


class Activation_ReLU: 

    def forward(self, inputs): 
        self.output = np.maximum(0, inputs) # azzera i negativi


class Activation_Softmax: 

    def forward(self, inputs): 
        exp_values    = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True)) # input - il massimo input per riga, per evitare overflow
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output   = probabilities


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
