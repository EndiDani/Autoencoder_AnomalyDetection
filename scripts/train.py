from models.model         import Model
from data.mnist           import load_mnist, load_saved_data, save_data
import matplotlib.pyplot  as plt
import numpy              as np

X_train, _, X_test, _ = load_saved_data()

loaded_model = Model.load(f"./results/model_checkpoints/autoencoder_final_model.pk1")
loaded_model.train(X_train, epochs = 30, batch_size = 64, print_every = 3, validation_data = X_test)

loaded_model.save(f"./results/model_checkpoints/autoencoder_final_model.pk1")


