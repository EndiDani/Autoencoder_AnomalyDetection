from   models.model      import Model
from   data.mnist        import load_saved_data
import matplotlib.pyplot as plt

_, _, X_test, _ = load_saved_data()

model = Model.load(f"./results/model_checkpoints/autoencoder_final_model.pk1")

loss  = model.evaluate(X_test)
print(f"Test Loss: {loss}")

plt.plot(model.lossHistory)  
plt.title("Loss during Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
