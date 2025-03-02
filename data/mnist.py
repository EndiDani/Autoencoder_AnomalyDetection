import tensorflow as tf
import numpy as np

def load_mnist():
    # Load del dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizzazione dei dati (0-255 -> 0-1)
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Remodeling delle immagini per aggiungere un canale -> 28x28x1
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    return (X_train, y_train), (X_test, y_test)

def save_data(X_train, y_train, X_test, y_test):
    np.savez_compressed('data/saved_data/mnist_normalized.npz', X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

def load_saved_data():
    data = np.load('data/saved_data/mnist_normalized.npz')
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']
