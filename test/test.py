import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Test code to verify TensorFlow and Keras are installed
# Also used to play with features of both of the above

NAME = "mnist-128X128-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Load MNIST
mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Normalize data
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

# Build model
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))

# Hyper params
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(xTrain, yTrain, epochs=3, batch_size=32, callbacks=[tensorboard])

# Accuracy
valLoss, valAccuracy = model.evaluate(xTest, yTest)
print (valLoss, valAccuracy)