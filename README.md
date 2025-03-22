# Autoencoder Anomaly Detection

Welcome to my project dedicated to **Anomaly Detection** using an **Autoencoder**, a neural network model I built **from scratch**!  
This is my first experience working with neural networks, and I decided to apply it to an **anomaly detection** problem using a real dataset. The goal is to build a system capable of identifying anomalies in a dataset, using the autoencoder as the main tool.

The autoencoder I developed was built entirely from scratch, without relying on high-level libraries like TensorFlow or Keras. The network is designed to learn how to reconstruct the input data — such as images — by encoding them into a lower-dimensional representation and then decoding them back to their original form. Once trained, the autoencoder can evaluate new data: if the reconstruction error (the difference between the input and the reconstructed output) is high, the input is considered an anomaly. This happens because the model was not able to reconstruct it well based on what it learned from normal data, effectively flagging it as an anomaly.

## Project Structure

The project is organized in a simple yet functional way to guide you step by step through the process. Here’s what you will find:

- **`data/`**: This folder contains the datasets used for training and evaluating the model. For this project, I chose the well-known **MNIST** dataset from TensorFlow, which contains images of handwritten digits, ideal for testing the anomaly detection capabilities of the autoencoder.

- **`models/`**: This folder includes all the scripts for defining and training the autoencoder. The core of the model, designed to "learn" from the data and then detect any anomalies, is located here.

- **`notebooks/`**: If you want to see the model in action or delve deeper into the architecture of the neural network, this folder contains the original version of the network, explained and visualized step by step.

- **`scripts/`**: This folder includes scripts to run training, evaluation, and analyze the **loss history** (error function). You can use them to fine-tune the model and monitor its performance over time.

- **`results/`**: This folder stores the model **checkpoints** and other results obtained during training. It’s essential for tracking progress and results.

## Requirements

All the dependencies are listed in the `requirements.txt` file.

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/EndiDani/Autoencoder_AnomalyDetection.git
   cd Autoencoder_AnomalyDetection
   ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Try the model!**
    ```bash
    python3 scripts/predict.py 
    ```

## References & Acknowledgements

This neural network was built by following the course on YouTube and the GitHub repository provided by **Sentdex**. A big thanks to Sentdex for the excellent practical and theoretical explanations! 

### Key Resources:

- **YouTube Course by Sentdex**: A comprehensive video series on neural networks, providing both theoretical and practical insights.
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)

- **GitHub Repository by Sentdex**: The GitHub repository accompanying the YouTube course, which contains code examples and additional resources.
  - [GitHub Repository](https://github.com/Sentdex/nnfs_book)

- **"Neural Networks and Deep Learning" by Michael Nielsen**: A fundamental book that helped me understand the theoretical aspects of neural networks, especially the backpropagation algorithm and network architecture.
  - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)

### Additional Learning Resources:

- **Autoencoder Architecture**: I built the autoencoder from scratch, which includes designing both the encoder and decoder parts of the network. This helped me understand how autoencoders can be used for anomaly detection by reconstructing input data and measuring the reconstruction error.

- **Activation Functions**: The autoencoder uses activation functions such as **ReLU** to introduce non-linearity in the model, enabling it to learn more complex patterns in the data.

- **Backpropagation**: The backpropagation algorithm, which updates the weights of the network based on the error between the predicted and actual output, was key in training the model effectively.

- **Optimization with Adam**: I utilized the **Adam optimizer** to minimize the reconstruction error during training. Adam is an adaptive optimizer that dynamically adjusts the learning rate.

- **Reconstruction Error for Anomaly Detection**: The concept of **reconstruction error** is central to this project. If the reconstruction error is high, the input data is flagged as an anomaly, indicating that the model could not efficiently reconstruct the input from the learned representation.

- **Overfitting and Underfitting**: Throughout the training process, I learned how to manage the balance between overfitting (when the model memorizes the training data) and underfitting (when the model doesn't learn enough from the training data).

- **Regularization Techniques**: Regularization methods such as **L2 regularization** can be useful in preventing the model from overfitting by penalizing large weights, improving the model's ability to generalize.

- **Training and Validation Split**: I used a training and validation split to ensure that the model learned to generalize better on unseen data, preventing overfitting and improving its performance.

- **Loss Functions**: The loss function used in this project is the **Mean Squared Error (MSE)** between the input and reconstructed output. This helps in measuring the quality of the reconstruction and is used to guide the model's optimization during training.

- **Evaluation Metrics**: After training, the model's performance is evaluated by checking the reconstruction errors on new, unseen data. This evaluation is essential in understanding how well the model generalizes and detects anomalies.
