import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

# Load MNIST
'''
    TrainingX data must look like this as a 3D vector
    [
        ex 1: [
            [first input], [second input], ..., [last input]
        ],
        ex 2: [
            [first input], [second input], ..., [last input]
        ],
        ...
        ex n: [
            [first input], [second input], ..., [last input]
        ]
    ]

    TrainingY data must be the following:
    [
        ex 1: [expected output 1],
        ex 2: [expected output 2],
        ...
        ex n: [expected output n]
    ]
'''
xTrain = np.asarray([[[0,0,0],[1,1,1]], [[0,1,0],[1,1,1]], [[1,0,1],[1,1,1]], [[1,1,1],[1,1,1]], [[1,1,1],[0,1,0]]])
yTrain = np.asarray([[1], [1], [1], [1], [0]])

xTest = np.asarray([[[0,0,0],[0,0,1]]])
yTest = np.asarray([[0]])

# Normalize data
# xTrain = tf.keras.utils.normalize(xTrain, axis=1)
# xTest = tf.keras.utils.normalize(xTest, axis=1)

# Build Model
model = Sequential()
model.add( LSTM(128, input_shape=(xTrain.shape[1:]), activation='relu', return_sequences=True) )
# model.add( CuDNNLSTM(128, input_shape=(xTrain.shape[1:]), return_sequences=True) )    If running with GPU
model.add( Dropout(0.2) )

model.add( LSTM(128, activation='relu', return_sequences=False) )
# model.add( CuDNNLSTM(128) )   If running with GPU
model.add( Dropout(0.2) )

model.add( Dense(32, activation='relu') )
model.add( Dropout(0.2) )

model.add( Dense(10, activation='softmax') )

# Hyper params
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss="sparse_categorical_crossentropy",
    optimizer=opt,
    metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='test/logs/{}'.format("testing-testing2"))

model.fit(xTrain, yTrain, epochs=3, callbacks=[tensorboard], validation_data=(xTest, yTest))