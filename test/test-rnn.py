import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, CuDNNLSTM

# Load MNIST
mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Normalize data
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

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

model.fit(xTrain, yTrain, epochs=3, validation_data=(xTest, yTest))