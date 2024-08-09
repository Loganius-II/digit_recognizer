'''
see: https://www.youtube.com/watch?v=Zi4i7Q0zrBs&t=95s
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Dataset with many different handwritten
# digits
mnist = tf.keras.datasets.mnist

# Training Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scaling and Greyscaling images
# the Y is labels so we don't
# want to grayscale that
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Using models
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# How many neurons
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# scales values down to one so we can use percentages
# of accuracy
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# For training the model
model.fit(x_train, y_train, epochs=5)
accuracy, loss = model.evaluate(x_test, y_test)

model.save('digits.keras')

for x in range(1,7):
    # Still need to understand. I only half know 
    img = cv2.imread(f'./digits/{x}.png')[:,:,0]
    #                            ^^^^^^
    img = cv2.resize(img, (28, 28))
    img = np.invert(np.array([img]))

    # Prediction:
    prediction = model.predict(img)
    print("Prediction:")
    print(np.argmax(prediction))

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()