from activation_functions import Activation_Sigmoid, Activation_Linear, Activation_ReLU
from layers               import Layer_Dense, Layer_Input
from loss                 import Loss, Loss_MeanSquaredError
from model                import Model
from optimizer            import Optimizer_Adam
from data.mnist           import load_mnist, load_saved_data, save_data
import matplotlib.pyplot as plt
import numpy as np

# Caricamento dati prima volta
#(X_train, y_train), (X_test, y_test) = load_mnist()
#save_data(X_train, y_train, X_test, y_test)

# Seconda volta
X_train, _, X_test, _= load_saved_data()

EncoderDense1 = Layer_Dense(784, 392)
Activation1   = Activation_ReLU()
EncoderDense2 = Layer_Dense(392, 196)
Activation2   = Activation_ReLU()
EncoderDense3 = Layer_Dense(196, 98)
Activation3   = Activation_ReLU()
EncoderDense4 = Layer_Dense(98, 49)
Activation4   = Activation_ReLU()

BottleneckDense = Layer_Dense(49, 49)
Activation5     = Activation_Sigmoid()

DecoderDense1 = Layer_Dense(49, 98)
Activation6   = Activation_Sigmoid()
DecoderDense2 = Layer_Dense(98, 196)
Activation7   = Activation_Sigmoid()
DecoderDense3 = Layer_Dense(196, 392)
Activation8   = Activation_Sigmoid()
DecoderDense4 = Layer_Dense(392, 784)
Activation9   = Activation_Sigmoid()

Model = Model()

Model.add(EncoderDense1)
Model.add(Activation1)
Model.add(EncoderDense2)
Model.add(Activation2)
Model.add(EncoderDense3)
Model.add(Activation3)
Model.add(EncoderDense4)
Model.add(Activation4)

Model.add(BottleneckDense)
Model.add(Activation5)

Model.add(DecoderDense1)
Model.add(Activation6)
Model.add(DecoderDense2)
Model.add(Activation7)
Model.add(DecoderDense3)
Model.add(Activation8)
Model.add(DecoderDense4)
Model.add(Activation9)

Model.finalize()

Model.train(X_train, epochs = 1000, print_every = 100, validation_data = X_test)

output = Model.forward(X_test, training = False)

num_images = 5  # Numero di immagini da visualizzare
indices = np.random.choice(X_test.shape[0], num_images, replace=False)

for idx in indices:
    # Ricostruisci le immagini a 28x28 per visualizzarle
    input_img = X_test[idx].reshape(28, 28)
    output_img = output[idx].reshape(28, 28)

    plt.figure(figsize=(4, 2))
    
    # Immagine di input
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')
    
    # Immagine ricostruita
    plt.subplot(1, 2, 2)
    plt.title("Ricostruzione")
    plt.imshow(output_img, cmap='gray')
    plt.axis('off')
    
    plt.show()
