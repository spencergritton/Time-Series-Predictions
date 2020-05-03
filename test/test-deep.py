import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import time

# Test code to verify TensorFlow and Keras are installed
# Also used to play with features of both of the above

# Load MNIST
mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Normalize data
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

dense_layers = [1]
layer_sizes = [64]

# Build model
model = Sequential()
model.add(Flatten())

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        NAME = "mnist-{}-{}".format( (((str(layer_size) + "X") * dense_layer)[:len(((str(layer_size) + "X") * dense_layer))-1]), int(time.time()))
        print("Starting model: {}".format(NAME))

        for layer in range(dense_layer):
            model.add(Dense(layer_size, activation=tf.nn.relu))

        model.add(Dense(10, activation=tf.nn.softmax))

        # Hyper Paramaters
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

        tensorboard = TensorBoard(log_dir='test/logs/{}'.format(NAME))
        model.fit(xTrain, yTrain, epochs=5, batch_size=32, validation_split=.2, callbacks=[tensorboard])
        # To use TensorBoard: python3 /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorboard/main.py --logdir='logs/'

        # Statistics
        # valLoss, valAccuracy = model.evaluate(xTest, yTest)
        # print ("Model: {}\nValidation Loss: {}\nValidation Accuracy: {}\n".format(NAME, valLoss, valAccuracy))