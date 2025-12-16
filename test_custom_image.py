# test your own handwritten digit image
# first train the model by running: python simple_mnist_nn_fixed.py
# then use this script to test custom images

import numpy as np
from simple_mnist_nn_fixed import train_network, predict_custom_image
import pandas as pd


print("training network...")
data = pd.read_csv('train.csv')
data = np.array(data)
np.random.shuffle(data)

data_train = data[1000:].T
Y_train = data_train[0].astype(np.int32)
X_train = data_train[1:].astype(np.float32)
X_train = X_train / 255.0

max_train_samples = 15000
if X_train.shape[1] > max_train_samples:
    X_train = X_train[:, :max_train_samples]
    Y_train = Y_train[:max_train_samples]

W1, b1, W2, b2 = train_network(X_train, Y_train, learning_rate=0.15, iterations=1000, hidden_size=128)
print("training complete!")

# test your image
# replace 'digit.png' with your image path
image_path = 'digit.jpeg'  

try:
    prediction = predict_custom_image(image_path, W1, b1, W2, b2)
    print(f"\nfinal prediction: {prediction}")
except FileNotFoundError:
    print(f"\nerror: image file '{image_path}' not found")
    
