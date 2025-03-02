import numpy as np
import pickle
import copy
from layers import Layer_Input


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
    
