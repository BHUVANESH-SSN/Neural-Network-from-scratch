
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


try:
    data = pd.read_csv('train.csv')
    print(f"Data loaded successfully: {data.shape}")
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please provide the MNIST dataset.")
    exit()

data = np.array(data)
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0].astype(np.int32)
X_dev = data_dev[1:].astype(np.float32)
X_dev = X_dev / 255.0

data_train = data[1000:].T
Y_train = data_train[0].astype(np.int32)
X_train = data_train[1:].astype(np.float32)
X_train = X_train / 255.0

max_train_samples = 15000
if X_train.shape[1] > max_train_samples:
    X_train = X_train[:, :max_train_samples]
    Y_train = Y_train[:max_train_samples]
    print(f"Using {max_train_samples} training samples for memory efficiency")

print(f"Training set: {X_train.shape[1]} examples")
print(f"Validation set: {X_dev.shape[1]} examples")



def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)



def initialize_parameters(hidden_size=128):
    W1 = (np.random.randn(hidden_size, 784) * np.sqrt(2.0 / 784)).astype(np.float32)
    b1 = np.zeros((hidden_size, 1), dtype=np.float32)
    W2 = (np.random.randn(10, hidden_size) * np.sqrt(2.0 / hidden_size)).astype(np.float32)
    b2 = np.zeros((10, 1), dtype=np.float32)
    return W1, b1, W2, b2



def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2



def one_hot_encode(Y):
    one_hot = np.zeros((10, Y.size))
    one_hot[Y, np.arange(Y.size)] = 1
    return one_hot


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def calculate_accuracy(predictions, Y):
    return np.mean(predictions == Y) * 100



def backward_propagation(Z1, A1, A2, W2, X, Y, m):
    one_hot_Y = one_hot_encode(Y)
    
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2



def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2



def train_network(X, Y, learning_rate=0.1, iterations=500, hidden_size=128):
    m = X.shape[1]
    W1, b1, W2, b2 = initialize_parameters(hidden_size)
    
    print(f"\nStarting training for {iterations} iterations...")
    print(f"Learning rate: {learning_rate}\n")
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, A2, W2, X, Y, m)
        
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if i % 50 == 0:
            predictions = get_predictions(A2)
            accuracy = calculate_accuracy(predictions, Y)
            print(f"Iteration {i:3d}: Accuracy = {accuracy:.2f}%")
    
    print("\nTraining complete!")
    return W1, b1, W2, b2



def visualize_network_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    input_x, hidden_x, output_x = 1.5, 5, 8.5
    
    input_neurons = 10
    input_y_positions = np.linspace(1, 9, input_neurons)
    ax.text(input_x, 9.5, 'Input Layer\n(784 pixels)', ha='center', fontsize=12, 
            weight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    for i, y in enumerate(input_y_positions):
        circle = plt.Circle((input_x, y), 0.15, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        if i == input_neurons // 2:
            ax.text(input_x - 1.2, y, '... 784 neurons ...', fontsize=9, va='center', weight='bold', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    hidden_neurons = 10
    hidden_y_positions = np.linspace(1, 9, hidden_neurons)
    ax.text(hidden_x, 9.5, 'Hidden Layer\n(128 neurons)\nReLU Activation', ha='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    for i, y in enumerate(hidden_y_positions):
        circle = plt.Circle((hidden_x, y), 0.2, color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(hidden_x, y, f'{i}', ha='center', va='center', fontsize=9, weight='bold')
    
    output_neurons = 10
    output_y_positions = np.linspace(1, 9, output_neurons)
    ax.text(output_x, 9.5, 'Output Layer\n(10 classes)\nSoftmax', ha='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    for i, y in enumerate(output_y_positions):
        circle = plt.Circle((output_x, y), 0.2, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(output_x, y, f'{i}', ha='center', va='center', fontsize=9, weight='bold')
    
    b1_y = 0.5
    bias1_circle = plt.Circle((input_x, b1_y), 0.18, color='yellow', ec='darkred', linewidth=2.5)
    ax.add_patch(bias1_circle)
    ax.text(input_x, b1_y, 'b1', ha='center', va='center', fontsize=9, weight='bold', color='darkred')
    ax.text(input_x, b1_y - 0.4, '(128×1)', ha='center', fontsize=7, style='italic')
    
    b2_y = 0.5
    bias2_circle = plt.Circle((hidden_x, b2_y), 0.18, color='yellow', ec='darkred', linewidth=2.5)
    ax.add_patch(bias2_circle)
    ax.text(hidden_x, b2_y, 'b2', ha='center', va='center', fontsize=9, weight='bold', color='darkred')
    ax.text(hidden_x, b2_y - 0.4, '(10×1)', ha='center', fontsize=7, style='italic')
    
    for in_y in input_y_positions:
        for hid_y in hidden_y_positions:
            ax.plot([input_x + 0.15, hidden_x - 0.2], [in_y, hid_y], 
                   'gray', alpha=0.25, linewidth=0.6)
    
    for hid_y in hidden_y_positions:
        ax.plot([input_x + 0.18, hidden_x - 0.2], [b1_y, hid_y], 
               'gray', alpha=0.25, linewidth=0.6, linestyle='-')
    
    ax.text((input_x + hidden_x) / 2, 9.2, 'FULLY CONNECTED\n(All 784 → All 128)', 
            ha='center', fontsize=9, style='italic', color='darkblue', weight='bold')
    
    for hid_y in hidden_y_positions:
        for out_y in output_y_positions:
            ax.plot([hidden_x + 0.2, output_x - 0.2], [hid_y, out_y], 
                   'gray', alpha=0.25, linewidth=0.6)
    
    for out_y in output_y_positions:
        ax.plot([hidden_x + 0.18, output_x - 0.2], [b2_y, out_y], 
               'gray', alpha=0.25, linewidth=0.6, linestyle='-')
    
    ax.text((hidden_x + output_x) / 2, 9.2, 'FULLY CONNECTED\n(All 128 → All 10)', 
            ha='center', fontsize=9, style='italic', color='darkgreen', weight='bold')
    
    ax.text((input_x + hidden_x) / 2, 0.3, 'W1: (128 × 784)\nb1: (128 × 1)', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text((hidden_x + output_x) / 2, 0.3, 'W2: (10 × 128)\nb2: (10 × 1)', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.title('Neural Network Architecture for MNIST Digit Recognition', 
             fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def visualize_forward_propagation(X, W1, b1, W2, b2, sample_idx=0):
    X_sample = X[:, sample_idx:sample_idx+1]
    Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X_sample)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    input_image = X_sample.reshape(28, 28)
    ax1.imshow(input_image, cmap='gray')
    ax1.set_title('Input Image\n(784 pixels)', fontsize=12, weight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(W1[:, :100], cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title('Weights W1 (sample)\n(10 × 784)', fontsize=12, weight='bold')
    ax2.set_xlabel('Input neurons (showing 100)')
    ax2.set_ylabel('Hidden neurons')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    Z1_values = Z1.flatten()[:10]
    hidden_size = Z1.shape[0]
    bars3 = ax3.bar(range(10), Z1_values, color='steelblue', edgecolor='black')
    ax3.set_title(f'Hidden Layer Z1 (First 10/{hidden_size})\n(Before ReLU)', fontsize=12, weight='bold')
    ax3.set_xlabel('Neuron')
    ax3.set_ylabel('Value')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    A1_values = A1.flatten()[:10]
    bars4 = ax4.bar(range(10), A1_values, color='green', edgecolor='black')
    ax4.set_title(f'Hidden Layer A1 (First 10/{hidden_size})\n(After ReLU)', fontsize=12, weight='bold')
    ax4.set_xlabel('Neuron')
    ax4.set_ylabel('Activation')
    ax4.grid(axis='y', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(W2, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax5.set_title('Weights W2\n(10 × 10)', fontsize=12, weight='bold')
    ax5.set_xlabel('Hidden neurons')
    ax5.set_ylabel('Output neurons')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    Z2_values = Z2.flatten()
    bars6 = ax6.bar(range(10), Z2_values, color='orange', edgecolor='black')
    ax6.set_title('Output Layer Z2\n(Before Softmax)', fontsize=12, weight='bold')
    ax6.set_xlabel('Digit Class')
    ax6.set_ylabel('Value')
    ax6.grid(axis='y', alpha=0.3)
    
    ax7 = fig.add_subplot(gs[1, 2:4])
    A2_values = A2.flatten()
    colors = ['red' if i == np.argmax(A2_values) else 'coral' for i in range(10)]
    bars7 = ax7.bar(range(10), A2_values, color=colors, edgecolor='black', linewidth=2)
    ax7.set_title('Output Probabilities (After Softmax)\nPrediction: Digit {}'.format(
        np.argmax(A2_values)), fontsize=12, weight='bold')
    ax7.set_xlabel('Digit Class')
    ax7.set_ylabel('Probability')
    ax7.set_ylim([0, 1])
    ax7.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars7, A2_values)):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('off')
    ax8.set_xlim(0, 10)
    ax8.set_ylim(0, 2)
    
    boxes_x = [0.5, 2, 3.5, 5, 6.5, 8, 9.5]
    boxes_labels = ['Input\n(784)', 'W1·X+b1', 'ReLU', 'Hidden\n(10)', 'W2·A1+b2', 'Softmax', 'Output\n(10)']
    boxes_colors = ['lightblue', 'yellow', 'lightgreen', 'lightgreen', 'yellow', 'lightcoral', 'lightcoral']
    
    for x, label, color in zip(boxes_x, boxes_labels, boxes_colors):
        box = FancyBboxPatch((x-0.3, 0.5), 0.6, 1, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax8.add_patch(box)
        ax8.text(x, 1, label, ha='center', va='center', fontsize=10, weight='bold')
    
    for i in range(len(boxes_x) - 1):
        arrow = FancyArrowPatch((boxes_x[i]+0.3, 1), (boxes_x[i+1]-0.3, 1),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax8.add_patch(arrow)
    
    ax8.text(5, 0.1, 'Forward Propagation Flow', ha='center', fontsize=14, 
            weight='bold', style='italic')
    
    fig.suptitle('Forward Propagation Visualization', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def visualize_backpropagation(X, Y, W1, b1, W2, b2, sample_idx=0):
    X_sample = X[:, sample_idx:sample_idx+1]
    Y_sample = Y[sample_idx:sample_idx+1]
    
    Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X_sample)
    
    dW1, db1, dW2, db2 = backward_propagation(Z1, A1, A2, W2, X_sample, Y_sample, 1)
    
    one_hot_Y = one_hot_encode(Y_sample)
    error = A2 - one_hot_Y
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    prediction = np.argmax(A2)
    actual = Y_sample[0]
    
    A2_values = A2.flatten()
    colors = ['green' if i == actual else ('red' if i == prediction else 'lightgray') 
             for i in range(10)]
    bars1 = ax1.bar(range(10), A2_values, color=colors, edgecolor='black', linewidth=2)
    ax1.set_title(f'Prediction vs Actual\nPred: {prediction}, Actual: {actual}', 
                 fontsize=12, weight='bold')
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Probability')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    error_values = error.flatten()
    colors2 = ['red' if abs(val) > 0.1 else 'pink' for val in error_values]
    bars2 = ax2.bar(range(10), error_values, color=colors2, edgecolor='black')
    ax2.set_title('Output Error (dZ2)\n(Predicted - Actual)', fontsize=12, weight='bold')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Error')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(dW2, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    ax3.set_title('Gradient dW2\n(How to update W2)', fontsize=12, weight='bold')
    ax3.set_xlabel('Hidden neurons')
    ax3.set_ylabel('Output neurons')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[1, 0])
    db2_values = db2.flatten()
    bars4 = ax4.bar(range(10), db2_values, color='purple', edgecolor='black')
    ax4.set_title('Gradient db2\n(Bias update for output)', fontsize=12, weight='bold')
    ax4.set_xlabel('Output neuron')
    ax4.set_ylabel('Gradient')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(axis='y', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(dW1[:, :100], cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
    ax5.set_title('Gradient dW1 (sample)\n(How to update W1)', fontsize=12, weight='bold')
    ax5.set_xlabel('Input neurons (showing 100)')
    ax5.set_ylabel('Hidden neurons')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 2])
    db1_values = db1.flatten()[:10]
    hidden_size = db1.shape[0]
    bars6 = ax6.bar(range(10), db1_values, color='darkgreen', edgecolor='black')
    ax6.set_title(f'Gradient db1 (First 10/{hidden_size})\n(Bias update for hidden)', fontsize=12, weight='bold')
    ax6.set_xlabel('Hidden neuron')
    ax6.set_ylabel('Gradient')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.grid(axis='y', alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, :])
    gradient_names = ['dW1', 'db1', 'dW2', 'db2']
    gradient_mags = [
        np.linalg.norm(dW1),
        np.linalg.norm(db1),
        np.linalg.norm(dW2),
        np.linalg.norm(db2)
    ]
    bars7 = ax7.bar(gradient_names, gradient_mags, color=['green', 'darkgreen', 'purple', 'darkviolet'],
                   edgecolor='black', linewidth=2)
    ax7.set_title('Gradient Magnitudes (How much to update each parameter)', 
                 fontsize=12, weight='bold')
    ax7.set_ylabel('Magnitude (L2 norm)')
    ax7.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars7, gradient_mags):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    fig.suptitle('Backpropagation Visualization - Gradients Flow Backward', 
                fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def visualize_weight_updates(W1_before, W2_before, W1_after, W2_after):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    im1 = axes[0, 0].imshow(W1_before[:, :100], cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_title('W1 Before Update', fontsize=12, weight='bold')
    axes[0, 0].set_ylabel('Hidden neurons')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(W1_after[:, :100], cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0, 1].set_title('W1 After Update', fontsize=12, weight='bold')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    W1_diff = W1_after[:, :100] - W1_before[:, :100]
    im3 = axes[0, 2].imshow(W1_diff, cmap='seismic', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('W1 Change (After - Before)', fontsize=12, weight='bold')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    im4 = axes[1, 0].imshow(W2_before, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[1, 0].set_title('W2 Before Update', fontsize=12, weight='bold')
    axes[1, 0].set_xlabel('Hidden neurons')
    axes[1, 0].set_ylabel('Output neurons')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(W2_after, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_title('W2 After Update', fontsize=12, weight='bold')
    axes[1, 1].set_xlabel('Hidden neurons')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    W2_diff = W2_after - W2_before
    im6 = axes[1, 2].imshow(W2_diff, cmap='seismic', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[1, 2].set_title('W2 Change (After - Before)', fontsize=12, weight='bold')
    axes[1, 2].set_xlabel('Hidden neurons')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    fig.suptitle('Weight Updates After One Gradient Descent Step', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()



def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    return get_predictions(A2)


def visualize_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index:index+1]
    
    prediction = make_predictions(current_image, W1, b1, W2, b2)[0]
    actual_label = Y[index]
    
    print(f"\nPrediction: {prediction}")
    print(f"Actual Label: {actual_label}")
    print(f"{'✓ CORRECT' if prediction == actual_label else '✗ WRONG'}")
    
    image = current_image.reshape(28, 28) * 255
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {prediction}, Actual: {actual_label}")
    plt.axis('off')
    plt.show()


def predict_custom_image(image_path, W1, b1, W2, b2):
    try:
        from PIL import Image
    except ImportError:
        print("please install pillow: pip install pillow")
        return
    
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32)
    
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    img_array = img_array / 255.0
    img_array = img_array.reshape(784, 1)
    
    prediction = make_predictions(img_array, W1, b1, W2, b2)[0]
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, img_array)
    confidence = A2.flatten()[prediction] * 100
    
    print(f"\nprediction: {prediction}")
    print(f"confidence: {confidence:.2f}%")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    ax1.imshow(img_array.reshape(28, 28), cmap='gray')
    ax1.set_title('your image (28x28)')
    ax1.axis('off')
    
    probs = A2.flatten() * 100
    colors = ['green' if i == prediction else 'gray' for i in range(10)]
    bars = ax2.bar(range(10), probs, color=colors, edgecolor='black')
    ax2.set_title(f'predicted: {prediction} ({confidence:.1f}%)')
    ax2.set_xlabel('digit')
    ax2.set_ylabel('probability %')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, probs)):
        if val > 5:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return prediction



if __name__ == "__main__":
    print("\n" + "="*50)
    print("VISUALIZING NETWORK ARCHITECTURE")
    print("="*50)
    visualize_network_architecture()
    
    W1, b1, W2, b2 = train_network(X_train, Y_train, learning_rate=0.15, iterations=1000, hidden_size=128)
    
    print("\n" + "="*50)
    print("TRAINING SET EVALUATION")
    print("="*50)
    train_predictions = make_predictions(X_train, W1, b1, W2, b2)
    train_accuracy = calculate_accuracy(train_predictions, Y_train)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    
    print("\n" + "="*50)
    print("VALIDATION SET EVALUATION")
    print("="*50)
    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    dev_accuracy = calculate_accuracy(dev_predictions, Y_dev)
    print(f"Validation Accuracy: {dev_accuracy:.2f}%")
    
    print("\n" + "="*50)
    print("VISUALIZING FORWARD PROPAGATION")
    print("="*50)
    print("Showing what happens when input passes through the network...")
    visualize_forward_propagation(X_train, W1, b1, W2, b2, sample_idx=0)
    
    print("\n" + "="*50)
    print("VISUALIZING BACKPROPAGATION")
    print("="*50)
    print("Showing how gradients flow backward to update weights...")
    visualize_backpropagation(X_train, Y_train, W1, b1, W2, b2, sample_idx=0)
    
    print("\n" + "="*50)
    print("VISUALIZING WEIGHT UPDATES")
    print("="*50)
    print("Performing one gradient descent step to show weight changes...")
    
    W1_before = W1.copy()
    W2_before = W2.copy()
    
    X_batch = X_train[:, :10]
    Y_batch = Y_train[:10]
    Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X_batch)
    dW1, db1, dW2, db2 = backward_propagation(Z1, A1, A2, W2, X_batch, Y_batch, 10)
    W1_after, b1_after, W2_after, b2_after = update_parameters(
        W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.1)
    
    visualize_weight_updates(W1_before, W2_before, W1_after, W2_after)
    
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    for i in range(4):
        visualize_prediction(i, X_train, Y_train, W1, b1, W2, b2)
    
    print("\n" + "="*50)
    print("CUSTOM IMAGE PREDICTION")
    print("="*50)
    print("to test your own handwritten digit:")
    print("  predict_custom_image('your_image.png', W1, b1, W2, b2)")
   
