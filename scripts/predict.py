from   dotenv import load_dotenv
import os
import sys

load_dotenv()

sys.path.extend(os.getenv("PYTHONPATH", "").split(":"))

from models.model         import Model
from data.mnist           import load_mnist, load_saved_data, save_data
import matplotlib.pyplot  as plt
import numpy              as np

_, _, X_test, _ = load_saved_data()

model      = Model.load(f"./results/model_checkpoints/autoencoder_final_model.pk1")
output     = model.forward(X_test, training = False)
losses     = []
num_images = 10
indices    = np.random.choice(X_test.shape[0], num_images, replace = False)

for idx in indices:
    input_img  = X_test[idx].reshape(28, 28)
    output_img = output[idx].reshape(28, 28)
    
    # Calcolo loss per rilevare anomalie
    data_loss, regularization_loss = model.loss.calculate(output[idx], input_img.reshape(-1, 784), include_regularization = True)
    losses.append(data_loss + regularization_loss)
    
    plt.figure(figsize=(4, 2))

    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(input_img, cmap = 'gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(output_img, cmap = 'gray')
    plt.axis('off')

    plt.show()

# Rilevazione anomalie
mean_loss = np.mean(losses)
std_loss  = np.std(losses)

# Soglia per definire un immagine anomalia: 1 deviazione standard sopra la media
threshold = mean_loss + 1 * std_loss  

# Identifico le anomalie
anomalies = []
for idx, loss in zip(indices, losses):
    if loss > threshold:
        anomalies.append(idx)

# Stampo le anomalie rilevate
if anomalies:
    for idx in anomalies:
        input_img = X_test[idx].reshape(28, 28)

        plt.figure(figsize=(4, 2))
        plt.title("Anomalous image")
        plt.imshow(input_img, cmap = 'gray')
        plt.axis('off')  

        plt.show() 

# Stampo le loss di ricostruzione    
plt.hist(losses, bins = num_images, edgecolor = 'black')
plt.title("Losses")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()