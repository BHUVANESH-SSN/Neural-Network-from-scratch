# MNIST Neural Network from Scratch

A complete implementation of a neural network built from scratch using only NumPy to recognize handwritten digits from the MNIST dataset. This project includes comprehensive visualizations of the network architecture, forward propagation, backpropagation, and weight updates.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-1.19+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents
- [Features](#features)
- [Neural Network Architecture](#neural-network-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [ReLU Activation Unit](#relu-activation-unit)
- [Visualizations](#visualizations)
- [Testing Custom Handwritten Digits](#testing-custom-handwritten-digits)
- [Learning Resources](#learning-resources)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Features

- **Pure NumPy Implementation**: No deep learning frameworks - built from mathematical foundations
- **Comprehensive Visualizations**: 
  - Network architecture diagram
  - Forward propagation flow
  - Backpropagation gradients
  - Weight update visualization
  - Prediction examples with actual images
- **Educational Comments**: Every function includes detailed explanations of what's happening
- **High Accuracy**: Achieves 85-92% accuracy on MNIST dataset
- **Modular Design**: Clear separation of concerns with 11 distinct modules

## Neural Network Architecture

```
Input Layer (784 neurons) → Hidden Layer (128 neurons, ReLU) → Output Layer (10 neurons, Softmax)
```

- **Input**: 28×28 pixel images flattened to 784 values
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9) with Softmax activation
- **Training**: Gradient descent with backpropagation

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- pillow (For custom Images)

## Installation

1. **Clone the repository**
```bash
git clone [https://github.com/yourusername/mnist-neural-network.git](https://github.com/BHUVANESH-SSN/Neural-Network-from-scratch.git)
cd mnist-neural-network
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install numpy pandas matplotlib
```

4. **Download MNIST dataset**
   - Download `train.csv` from [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
   - Place it in the project directory

## Usage

**Basic Usage:**
```bash
python simple_mnist_nn_fixed.py
```

**Customize hyperparameters in the code:**
```python
# In the main execution section
W1, b1, W2, b2 = train_network(
    X_train, 
    Y_train, 
    learning_rate=0.15,    # Adjust learning rate
    iterations=1000,        # Number of training iterations
    hidden_size=128         # Number of hidden neurons
)
```

## Project Structure

```
NN_Scratch/
│
├── simple_mnist_nn_fixed.py  # Main neural network implementation
├── train.csv                 # MNIST training dataset
├── test.csv                  # MNIST test dataset (optional)
├── digit.jpeg                # HandWritten image
├── test_custom_image.py      # Test Your Own Custom Image 
├── sample_submission.csv     # Sample submission format
└── README.md                 # This file
```

## How It Works

### 1. **Data Loading & Preprocessing**
- Loads MNIST data from CSV
- Normalizes pixel values from [0, 255] to [0, 1]
- Converts to float32 for memory efficiency
- Splits into training (15,000) and validation (1,000) sets

### 2. **Forward Propagation**
```
Z1 = W1 · X + b1
A1 = ReLU(Z1)
Z2 = W2 · A1 + b2
A2 = Softmax(Z2)
```

### 3. **Backpropagation**
Calculates gradients using chain rule:
```
dZ2 = A2 - Y
dW2 = (1/m) · dZ2 · A1ᵀ
dZ1 = W2ᵀ · dZ2 · ReLU'(Z1)
dW1 = (1/m) · dZ1 · Xᵀ
```

### 4. **Gradient Descent**
Updates weights to minimize error:
```
W = W - α · dW
b = b - α · db
```

## Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 90-95% |
| **Validation Accuracy** | 85-90% |
| **Training Set Size** | 15,000 samples |
| **Training Time** | ~30-50 seconds |
| **Iterations** | 1000 |
| **Learning Rate** | 0.15 |
| **Hidden Neurons** | 128 |
| **Data Type** | float32 (memory optimized) |
<img width="580" height="527" alt="image" src="https://github.com/user-attachments/assets/0959b1be-ef16-45f8-ab93-b03fdbf17ba0" />

### Performance Tips

**To increase accuracy:**
- Increase `max_train_samples` to 20,000-25,000 (if RAM allows)
- Increase `hidden_size` to 256 (may need more RAM)
- Increase `iterations` to 1500-2000
- Try `learning_rate` between 0.1-0.2
- Add a second hidden layer

**To reduce training time:**
- Decrease `iterations` to 500
- Decrease `hidden_size` to 64
- Reduce `max_train_samples` to 10,000

**Memory optimizations (already implemented):**
-  float32 data type (50% memory savings)
-  Limited training samples (prevents overflow)
-  He initialization for efficient training
## ReLU Activation Unit
<img width="988" height="563" alt="image" src="https://github.com/user-attachments/assets/c80ea603-0653-4ec6-841b-90a1f81dbfd8" />

![ReLU](https://media.geeksforgeeks.org/wp-content/uploads/20250129162127770664/Relu-activation-function.png)
## Visualizations

The code generates multiple visualizations:

1. **Network Architecture**: Visual diagram of the network structure
 <img width="1910" height="947" alt="image" src="https://github.com/user-attachments/assets/d3096754-051c-46c4-b468-1c47c97e7f83" />

2. **Forward Propagation**: Step-by-step data flow through the network
<img width="1912" height="921" alt="image" src="https://github.com/user-attachments/assets/d7f609d5-d52e-4b1d-9f76-b8c522468767" />

3. **Backpropagation**: Gradient calculation and error propagation
<img width="1776" height="932" alt="image" src="https://github.com/user-attachments/assets/f434bfa9-187f-42ee-96ea-f4c89701a529" />

4. **Weight Updates**: Before/after comparison of weight matrices
<img width="1898" height="950" alt="image" src="https://github.com/user-attachments/assets/4b6c0ddb-a64e-41b1-980c-57c64b13f966" />

5. **Predictions**: Individual digit predictions with confidence scores
<img width="372" height="373" alt="image" src="https://github.com/user-attachments/assets/dd164897-f9dd-4fac-a30a-357a75327813" />
<img width="367" height="372" alt="image" src="https://github.com/user-attachments/assets/cdb51017-0875-49c7-8bb2-9c40a6e9f8f1" />
<img width="352" height="357" alt="image" src="https://github.com/user-attachments/assets/ce7483fd-2393-4954-896e-c144918ef56e" />
<img width="362" height="365" alt="image" src="https://github.com/user-attachments/assets/3d9edf03-2559-4060-b2db-0312b3a6c109" />




## Testing Custom Handwritten Digits

### How Pillow Works

Pillow (PIL) is a python image processing library that helps us work with custom images:

**image processing steps:**
1. **load image** - reads image files (png, jpg, bmp, etc.)
2. **convert to grayscale** - converts color images to black and white
3. **resize** - scales image to 28x28 pixels to match mnist format
4. **convert to array** - transforms image into numpy array for processing
5. **normalize** - scales pixel values from 0-255 to 0-1 range

### Using Custom Images

**installation:**
```bash
pip install pillow
```
## Image requirements:

1. single digit centered in image
2. white background with black digit (or vice versa)
3. clear and bold handwriting
4. any size (will be resized to 28x28)
5. supported formats: jpg, png, bmp, gif
## Example Test Results
test image 1: handwritten "8"
![8 Image](/digit.jpeg)
## Test Prediction Results
<img width="1600" height="761" alt="image" src="https://github.com/user-attachments/assets/09e88d0d-6787-4992-bfc2-58ca7425f5a2" />

## Learning Resources

**Key Concepts Implemented:**
- Neural network fundamentals
- Activation functions (ReLU, Softmax)
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Weight initialization (He initialization)
- Loss calculation (Cross-entropy)


## Customization

### Adjust Network Architecture
```python
def initialize_parameters(hidden_size=256):  # Change hidden layer size
    # ... rest of code
```

### Modify Training Parameters
```python
train_network(
    X_train, 
    Y_train, 
    learning_rate=0.2,      # Higher = faster but less stable
    iterations=2000,         # More = better accuracy
    hidden_size=256          # More = higher capacity
)
```

### Add More Layers
You can extend the network to have multiple hidden layers by modifying the forward and backward propagation functions.

## Troubleshooting

**Low Accuracy (<75%)?**
- Increase `hidden_size` to 128 or 256
- Increase `iterations` to 1000+
- Check if data is normalized properly

**Code runs but no visualizations?**
- Ensure matplotlib backend is configured
- Try running in Jupyter Notebook
- Check if `plt.show()` is being called

**Memory errors?**
- Already optimized: Uses float32 instead of float64
- Training set limited to 15,000 samples by default
- To use more data: Increase `max_train_samples` in code (line ~48)
- Decrease `hidden_size` if still facing issues
- Close other memory-intensive applications

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or suggestions, please open an issue on GitHub.

---

