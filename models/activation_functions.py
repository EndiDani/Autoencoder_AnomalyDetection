import numpy as np


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


class Activation_Linear: 

    def forward(self, inputs, training): 
        self.inputs = inputs # Copia valore
        self.output = inputs

    def backward(self, dvalues):
        # Derivata e' 1 che moltiplicato per dvalue rimane invariato (chain rule)
        self.dinputs = dvalues.copy()

    def predictions(self, outputs): 
        return (outputs > 0.5) * 1
    

# Aggiungo una nuova attivazione: la Sigmoidea
# Questo perché i dati che tratterò sono normalizzati nell'intervallo [0, 1],
# che è l'intervallo in cui la sigmoide restituisce valori.
class Activation_Sigmoid:

    def forward(self, inputs, training):
        self.inputs  = inputs # Salvataggio inputs 
        self.output  = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
