import os
import os.path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pickle
import time
import random
import pandas as pd
import gc

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional, Input, Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adadelta
from tensorflow.keras import regularizers

# File to plot predictions of the BiDirectional-LSTM-GRU network
dataset="coinbase_180_30"
pickle_in = open(f"{dataset}.pickle", "rb")
data = pickle.load(pickle_in)
pickle_in.close()

train, test, parameters = data
trainX, trainY = train
testX, testY = test

model_name = 'Bidirectional-LSTM-GRU-250'

# Hyper params
DROPOUT = 0.5
EPOCHS = 250
BATCH_SIZE = 256
OPTIMIZER = Adadelta(learning_rate=1.0, rho=0.95)
optimizer_str = 'Adadelta: learning_rate=1.0, rho=0.95'
REGULARIZATION = regularizers.l2(0.01)
regularization_str = 'l2: 0.01, output penalty "activity"'

# Dataset data
INPUT_LEN = parameters['input_len']
OUTPUT_LEN = parameters['output_len']
standardization = parameters['standardization']

model = Sequential()
model.add( Bidirectional(LSTM(INPUT_LEN, input_shape=(trainX.shape[1:]), return_sequences=True, activity_regularizer=REGULARIZATION)) )
model.add( Dropout(DROPOUT) )
model.add( BatchNormalization() )

model.add( Bidirectional(GRU(INPUT_LEN, return_sequences=False, activity_regularizer=REGULARIZATION)) )
model.add( Dropout(DROPOUT) )
model.add( BatchNormalization() )

model.add( Dense(1800, activation='relu', activity_regularizer=REGULARIZATION) )
model.add( Dropout(DROPOUT) )
model.add( BatchNormalization() )

model.add( Dense(INPUT_LEN, activation='relu', activity_regularizer=REGULARIZATION) )
model.add( Dropout(DROPOUT) )
model.add( BatchNormalization() )

model.add( Dense(OUTPUT_LEN, activation='linear') )

model.compile(loss="mse",
                optimizer=OPTIMIZER,
                metrics=['mae'])

model.build(trainX.shape)
model.load_weights('experiment-3/checkpoints/Bidirectional-LSTM-GRU-250.h5')

value_to_predict = 1500

sampleX = np.asarray([testX[value_to_predict]]) # Input
sampleXClosingValue = sampleX[0][1] # Input closing values
sampleY = testY[value_to_predict] # Output closing values truth
prediction = model.predict(sampleX)[0] # predicted output closing values

# Plot prediction
fig,ax=plt.subplots()

# Primary Axis Labels
ax.set_xlabel("Time")
ax.set_ylabel("Closing Price")

# Primary axis data
ax.plot( sampleY, color="orange" )
ax.plot( prediction, color="blue" )

# Legend
loss_patch = mpatches.Patch(color='orange', label='True Output')
val_loss_patch = mpatches.Patch(color='blue', label='Predictions')
plt.legend(handles=[loss_patch, val_loss_patch])
plt.rcParams["legend.fontsize"] = 12

plt.title(model_name, loc='center')

plt.show()