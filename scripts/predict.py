from models.model         import Model
from data.mnist           import load_mnist, load_saved_data, save_data
import matplotlib.pyplot  as plt
import numpy              as np

_, _, X_test, _ = load_saved_data()

model  = Model.load(f"./results/model_checkpoints/autoencoder_final_model.pk1")

output = model.forward(X_test, training = False)

num_images = 10
indices    = np.random.choice(X_test.shape[0], num_images, replace = False)

for idx in indices:
    input_img  = X_test[idx].reshape(28, 28)
    output_img = output[idx].reshape(28, 28)

    plt.figure(figsize=(4, 2))

    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Ricostruzione")
    plt.imshow(output_img, cmap='gray')
    plt.axis('off')

    plt.show()
