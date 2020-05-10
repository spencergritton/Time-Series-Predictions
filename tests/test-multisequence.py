import pandas as pd
from datetime import datetime
import time
import os
import numpy as np
from collections import deque
import random
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Bidirectional, Input, Dropout, BatchNormalization, TimeDistributed
from keras.layers import Layer, InputSpec
from keras.callbacks import TensorBoard, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adadelta
from keras import regularizers


# Set reproducable seed values to compare each experiment based on their outputs and not seed values
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

np.random.seed(42)
random.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# Model and Dataset parameters
SEQUENCE_LENGTH = 2
NUMBER_MINUTES_TO_PREDICT = 2
FILE_TO_PREDICT = "bitcoin-historical-data/coinbase_test"

TRAIN_PERCENT = .85

# Import data into one df with columns: Volume, Close, Low, High, Open, Date
def createMainDf(file):
    dataset = f"data/{file}.csv"
    df = pd.read_csv(dataset)

    df = df[["Date", "Open", "High", "Low", "Close", "Volume_BTC", "Volume_Dollars", "Weighted_Price"]]

    df = cleanData(df)

    df = df.astype({f"Volume_BTC": float})
    df = df.astype({f"Volume_Dollars": float})
    df = df.astype({f"Weighted_Price": float})
    df = df.astype({f"Close": float})
    df = df.astype({f"Low": float})
    df = df.astype({f"High": float})
    df = df.astype({f"Open": float})
    df = df.astype({f"Date": int})

    # Sort by Date
    df.sort_values("Date", inplace=True, ascending=True)

    return cleanData(df)

def cleanData(df):
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep].astype(np.float64)
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)
    return df

def reduceTargets(row):
    if row['Index'] + 1 + NUMBER_MINUTES_TO_PREDICT > len(row['Target']):
        return "na"
    return row['Target'][row['Index'] + 1 : row['Index'] + 1 + NUMBER_MINUTES_TO_PREDICT]

def addTargets(df):
    # Set temporary index column for setting targets
    df["Index"] = [ *range(0, len( df.index.values) ) ]

    # Set each target column to maximum amount of targets
    df['Target'] = [df['Close'].values.tolist()] * len(df['Close'].values.tolist())
    # Reduce the targets to the correct values
    df['Target'] = df.apply(reduceTargets, axis=1)
    # Remove invalid rows
    df = df[df.Target != "na"]

    return df

# Split df into training and testing set. Last (1-TRAIN_PERCENT)% of data is in test
def splitTrainAndTest(df):
    dateValues = df["Date"].values
    last_x_pct = dateValues[ -int((1-TRAIN_PERCENT) * len(dateValues)) ]
    
    train = df[df.Date < last_x_pct]
    test = df[df.Date >= last_x_pct]

    return train, test

# Normalize the training set, output mean and std to use in normalizing test / predictions
def normalizeTrain(df):
    result = df.copy()
    standardization = {} # Map of standardized values for each column

    for feature_name in df.columns:
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        standardization[feature_name] = mean, std

        result[feature_name] = (df[feature_name] - mean) / std

    result.fillna(mean, inplace=True)

    return result, standardization

# Normalize test set using mean and std of training set
def normalizeTest(df, standardization):
    result = df.copy()
    for feature_name in df.columns:
        mean, std = standardization[feature_name]
        result[feature_name] = (df[feature_name] - mean) / std

    result.fillna(mean, inplace=True)

    return result

def reduceListSize(list, index):
    return list[index-SEQUENCE_LENGTH : index]

def reduceSequences(row):
    if row['Index'] < SEQUENCE_LENGTH:
        return "na"
    sequence = row['InputSequence']
    result = [reduceListSize(item, row['Index']) for item in sequence]
    return result

def addInputSequences(df):
    # Set temporary index column for setting targets
    df["Index"] = [ *range(0, len( df.index.values) ) ]

    # Set each row in InputSequence column to be a list of all values from each column
    df['InputSequence'] = [ 
        [
            df['Open'].values.tolist(),
            df['High'].values.tolist(),
            df['Low'].values.tolist(),
            df['Close'].values.tolist(),
            df['Volume_BTC'].values.tolist(),
            df['Volume_Dollars'].values.tolist(),
            df['Weighted_Price'].values.tolist(),
        ]
    ] * len(df['Index'].values.tolist())

    # Reduce the input sequences to the correct values
    df['InputSequence'] = df.apply(reduceSequences, axis=1)
    # Remove invalid rows
    df = df[df.InputSequence != "na"]

    return df

df = createMainDf(FILE_TO_PREDICT)

# Split into training and testing dataframe
trainDf, testDf = splitTrainAndTest(df)

# Drop date column
trainDf = trainDf.drop('Date', axis=1)
testDf = testDf.drop('Date', axis=1)

# Normalize training, normalize testing based on training normalization
trainDf, standardization = normalizeTrain(trainDf)
testDf = normalizeTest(testDf, standardization)

# Add target sequences
trainDf = addTargets(trainDf)
testDf = addTargets(testDf)

trainDf = addInputSequences(trainDf)
testDf = addInputSequences(testDf)

# Drop extra columns
trainDf = trainDf.drop('Index', axis=1)
testDf = testDf.drop('Index', axis=1)
trainDf = trainDf.drop('High', axis=1)
testDf = testDf.drop('High', axis=1)
trainDf = trainDf.drop('Low', axis=1)
testDf = testDf.drop('Low', axis=1)
trainDf = trainDf.drop('Close', axis=1)
testDf = testDf.drop('Close', axis=1)
trainDf = trainDf.drop('Open', axis=1)
testDf = testDf.drop('Open', axis=1)
trainDf = trainDf.drop('Volume_BTC', axis=1)
testDf = testDf.drop('Volume_BTC', axis=1)
trainDf = trainDf.drop('Volume_Dollars', axis=1)
testDf = testDf.drop('Volume_Dollars', axis=1)
trainDf = trainDf.drop('Weighted_Price', axis=1)
testDf = testDf.drop('Weighted_Price', axis=1)

# Get training and testing data in sequences of appropriate form
trainX = np.array( trainDf['InputSequence'].values.tolist() )
trainY = np.array( trainDf['Target'].values.tolist() )
testX = np.array( testDf['InputSequence'].values.tolist() )
testY = np.array( testDf['Target'].values.tolist() )

# Get training and testing data in sequences of appropriate form
# trainX, trainY = getXandY(trainDf)
# testX, testY = getXandY(testDf)

# End data set generation
# --------------------------------
# Start model generation

# Build the model
# Thanks to https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
# for showing how to make a MANY TO MANY model in Keras
model = Sequential()
model.add( LSTM(64, input_shape=(trainX.shape[1:]), return_sequences=False) )
model.add( Dropout(0.5) )
model.add( BatchNormalization() )

model.add( Dense(64, activation='relu', activity_regularizer=regularizers.l1(0.01)) )
model.add( Dropout(0.5) )
model.add( BatchNormalization() )

model.add( Dense(NUMBER_MINUTES_TO_PREDICT, activation='linear') )

model.compile(loss="mse",
                optimizer=Adadelta(),
                metrics=['mae'])

# File name
fileName = "test-LSTM-manytomany-64.h5"

# Callbacks
terminateOnNan = TerminateOnNaN()
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='min', baseline=None, restore_best_weights=True)
reduceOnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.001)
modelCheckpoint = ModelCheckpoint(f'checkpoints/{fileName}', monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir=f'logs/{fileName}')

# Train model
history = model.fit(
    trainX, trainY, 
    epochs=100, 
    validation_data=(testX, testY),
    batch_size=32,
    callbacks=[
        terminateOnNan,
        earlyStopping,
        reduceOnPlateau,
        modelCheckpoint,
    ],
)

# Generate chart
# Using this as assistance: 
# https://cmdlinetips.com/2019/10/how-to-make-a-plot-with-two-different-y-axis-in-python-with-matplotlib/

fig,ax=plt.subplots()

# Primary Axis Labels
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")

# Primary axis data
ax.plot( history.history['loss'], color="orange" )
ax.plot( history.history['val_loss'], color="red" )

# Secondary axis data and labels
ax2=ax.twinx()
ax2.plot( history.history['lr'], color="blue" )
ax2.set_ylabel(" Learning Rate" )

# Legend
loss_patch = mpatches.Patch(color='orange', label='Train Loss')
val_loss_patch = mpatches.Patch(color='red', label='Val Loss')
lr_patch = mpatches.Patch(color='blue', label='Learning Rate')
plt.legend(handles=[loss_patch, val_loss_patch, lr_patch])

plt.show()

score = model.evaluate(testX, testY, verbose=0)

print(model.metrics_names)
print(score)
print('Test loss:', score[0])
print('Test mae:', score[1])

# save the plot as a file
# fig.savefig('test.jpg', format='jpeg', dpi=250, bbox_inches='tight')