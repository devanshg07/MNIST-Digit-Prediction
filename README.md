# Digit Detection: 2-Layer Neural Network (MNIST)

This project trains a simple 2-layer neural network to classify handwritten digits from the MNIST dataset using PyTorch.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training and evaluation:**
   ```bash
   python mnist_train.py
   ```

The script will load the MNIST .ubyte files from the `MNIST Dataset` directory, train the model for 5 epochs, and print the test accuracy.

## Files
- `mnist_train.py`: Main training and evaluation script
- `requirements.txt`: Python dependencies
- `MNIST Dataset/`: Directory containing MNIST .ubyte files 