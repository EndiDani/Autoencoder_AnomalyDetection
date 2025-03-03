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

model = Model()

model.add(EncoderDense1)
model.add(Activation1)
model.add(EncoderDense2)
model.add(Activation2)
model.add(EncoderDense3)
model.add(Activation3)
model.add(EncoderDense4)
model.add(Activation4)

model.add(BottleneckDense)
model.add(Activation5)

model.add(DecoderDense1)
model.add(Activation6)
model.add(DecoderDense2)
model.add(Activation7)
model.add(DecoderDense3)
model.add(Activation8)
model.add(DecoderDense4)
model.add(Activation9)

model.finalize()

model.train(X_train, epochs = 1, batch_size = 64, print_every = 100, validation_data = X_test)

model.save((f"./results/model_checkpoints/autoencoder_final_model.pk1"))
