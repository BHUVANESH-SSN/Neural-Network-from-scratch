# MNIST Neural Network from Scratch

A complete implementation of a neural network built from scratch using only NumPy to recognize handwritten digits from the MNIST dataset. This project includes comprehensive visualizations of the network architecture, forward propagation, backpropagation, and weight updates.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-1.19+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents
- [Features](#features)
- [Neural Network Architecture](#neural-network-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Visualizations](#visualizations)
- [Learning Resources](#learning-resources)

## ✨ Features

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

## 🧠 Neural Network Architecture

```
Input Layer (784 neurons) → Hidden Layer (128 neurons, ReLU) → Output Layer (10 neurons, Softmax)
```

- **Input**: 28×28 pixel images flattened to 784 values
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9) with Softmax activation
- **Training**: Gradient descent with backpropagation

## 📦 Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mnist-neural-network.git
cd mnist-neural-network
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install numpy pandas matplotlib
```

4. **Download MNIST dataset**
   - Download `train.csv` from [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
   - Place it in the project directory

## 💻 Usage

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

## 📁 Project Structure

```
NN_Scratch/
│
├── simple_mnist_nn_fixed.py  # Main neural network implementation
├── train.csv                 # MNIST training dataset
├── test.csv                  # MNIST test dataset (optional)
├── sample_submission.csv     # Sample submission format
└── README.md                 # This file
```

## 🔍 How It Works

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

## 📊 Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 85-90% |
| **Validation Accuracy** | 83-88% |
| **Training Set Size** | 15,000 samples |
| **Training Time** | ~30-50 seconds |
| **Iterations** | 1000 |
| **Learning Rate** | 0.15 |
| **Hidden Neurons** | 128 |
| **Data Type** | float32 (memory optimized) |

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
## 🎨 Visualizations

The code generates multiple visualizations:

1. **Network Architecture**: Visual diagram of the network structure
2. <img width="1910" height="947" alt="image" src="https://github.com/user-attachments/assets/d3096754-051c-46c4-b468-1c47c97e7f83" />

3. **Forward Propagation**: Step-by-step data flow through the network
4. **Backpropagation**: Gradient calculation and error propagation
5. **Weight Updates**: Before/after comparison of weight matrices
6. **Predictions**: Individual digit predictions with confidence scores

## 📚 Learning Resources

**Key Concepts Implemented:**
- Neural network fundamentals
- Activation functions (ReLU, Softmax)
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Weight initialization (He initialization)
- Loss calculation (Cross-entropy)


## 🛠️ Customization

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

## 🐛 Troubleshooting

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

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

