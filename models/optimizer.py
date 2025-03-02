import numpy as np


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