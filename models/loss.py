import numpy as np


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


class Loss_MeanSquaredError(Loss): 

    def forward(self, y_pred, y_true): 
        sample_losses = np.mean((y_true - y_pred) ** 2, axis = -1)
        return sample_losses
    
    def backward(self, dvalues, y_true): 
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Calcolo del gradiente e normalizzazione
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples