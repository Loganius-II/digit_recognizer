'''
see: https://www.youtube.com/watch?v=w8yWXqWQYmU
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from PIL import Image
import PIL.ImageOps
mnist_test_filepath = './mnist_test.csv'
mnist_train_filepath = './mnist_train.csv'

data = pd.read_csv(mnist_train_filepath)
#data.head()

# convert csv to numpy array
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T # transposing it turns the rows into columns

X_dev = data_dev[1:n] # the grayscale pixel values of each image
Y_dev = data_dev[0] # the label (what number the image represents)
X_dev = X_dev / 255.
data_train = data[1000:m].T 
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
'''
Theoretically, all of the code above can be shortened to
the following code if tensorflow datasets are used:

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

But I don't want to use tensorflow,
I wan't it to be as primitive as possible
'''

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2

def relu(Z):
    # activation function
    # for forward propagation.
    # This is a common formula
    # in neural networks to
    # add nonlinearity to the 
    # inputs
    return np.maximum(0, Z) # if an element in Z > 0 then return Z
                            # else 0

'''
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
'''

def softmax(Z):
    Z -= np.max(Z)  # Subtract the max of Z to prevent overflow
    exp_values = np.exp(Z)
    return exp_values / np.sum(exp_values, axis=0)

def deriv_relu(Z):
    return Z > 0

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)    
    return Z1, A1, Z2, A2

def one_hot(Y):
    # one hot encoding
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# back propagation trains the model
# by the errors, so how off it was
# the onehot Y is a output
# the back prop formulas are nueral network
# formulas that are advanced so thats why
# they have wierd names
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

'''
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
'''
# This function updates the weights and biases 
# as per the formulas
def update_params(W1, b1, W2, b2,dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_acc(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def get_pred(A2):
    return np.argmax(A2, 0)

# Gradient descent is the term
# for training a neural network
# as a whole: forward and back
# propagation, as well as error
# handling, assesing, and correcting
# weights and biases
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Update info every 10 iterations
        if i % 10 == 0:
            os.system('cls')
            print(f"Iteration: {i}")
            print("Accuracy: ", get_acc(get_pred(A2), Y) )

    return W1, b1, W2, b2 

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

def make_prediction(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    prediction = get_pred(A2)
    return prediction
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_prediction(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print(f"Prediction: {prediction}")
    print(f"Label: {label}")
    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def predict_img(img, label, W1, b1, W2, b2):
    current_image = Image.open(img).convert('L')
    current_image = PIL.ImageOps.invert(current_image)
    current_image = current_image.resize((28,28))
    current_image = np.array(current_image) / 255.0
    current_image = current_image.reshape(784, 1)
    prediction = make_prediction(current_image, W1, b1, W2, b2)
    print(f"Prediction: {prediction}")
    print(f"Label: {label}")
    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

'''
test_prediction(5, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(6, W1, b1, W2, b2)
test_prediction(47, W1, b1, W2, b2)'''
'''
for i in range(1, 7):
    predict_img(f'{i}.png', i, W1, b1, W2, b2)
'''

'''
NOTE: test_prediction function uses preloaded 
handwritten digits, so it doesn't matter what you put
for the index param because it will always be different
because the pictures are shuffled every time. HOWEVER,
if you want to use your own pictures, just put: predict_img('path/to/your/image.png, W1, b1, W2, b2)
'''