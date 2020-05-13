import pandas as pd
from datetime import datetime
import time
import os
import numpy as np
from collections import deque
import random

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Bidirectional, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# This model will simply guess if the next price will be higher or lower than the current price
SEQUENCE_LENGTH = 30
FUTURE_MINUTE_TO_PREDICT = 10
FILE_TO_PREDICT = "BTC-USD"

TRAIN_PERCENT = .90

# Import and combine data into one dataframe
def createMainDf(file):
    dataset = f"data/{file}.csv"
    df = pd.read_csv(dataset)

    df.set_index("Date", inplace=True)
    df = df[["Close", "Volume"]]
    df = df.astype({f"Volume": float})
    df = df.astype({f"Close": float})

    # Sort by Date
    df.sort_index(inplace=True, ascending=True)

    return cleanData(df)

def cleanData(df):
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep].astype(np.float64)
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)
    return df

def addTargets(df):
    df['Target'] = df[f"Close"].shift(-FUTURE_MINUTE_TO_PREDICT)
    return cleanData(df)

# Split train and test code taken from: https://www.youtube.com/sentdex
def splitTrainAndTest(df):
    dateValues = df.index.values
    last_x_pct = dateValues[-int((1-TRAIN_PERCENT)*len(dateValues))]
    
    train = df[df.index < last_x_pct]
    test = df[df.index >= last_x_pct]
    return train, test

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean) / std
    result.fillna(mean, inplace=True)
    return result

# getXandY code partially taken from https://www.youtube.com/sentdex
def getXandY(df):
    df = normalize(df)
    
    sequential_data = []
    previous_days = deque(maxlen=SEQUENCE_LENGTH)

    for i in df.values:
        previous_days.append([n for n in i[:-1]])
        if len(previous_days) == SEQUENCE_LENGTH:
            sequential_data.append([np.array(previous_days), i[-1]])
    # Sequential data is of form
    '''
        [
            ex 1:[
                [ [c0, v0], [c1, v1], ... [cn, vn] ],
                target0
            ],
            ex 2: [
                [ [c1, v1], [c2, v2], ... [cn1, vn1] ],
                target1
            ]
        ]
    '''
    random.shuffle(sequential_data)

    # Create inputs and outputs
    X = []  # inputs
    y = []  # outputs

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.asarray(y)


# Start main code 
df = createMainDf(FILE_TO_PREDICT)
df = addTargets(df)
trainDf, testDf = splitTrainAndTest(df)

trainX, trainY = getXandY(trainDf)
testX, testY = getXandY(testDf)

# Model Building params for multiple models
LAYERS = [2, 3]
SIZE = [64, 128]
LEARNER = [tf.keras.optimizers.RMSprop()]

# Build each model and run for 5 epochs to get good idea of possibilities
for layer in LAYERS:
    for s in SIZE:
        for l in LEARNER:
            print(f"Building new model: layers:{layer} size:{s} learner:{l}")
            # Build Model
            model = Sequential()
            model.add( GRU(s, input_shape=(trainX.shape[1:]), return_sequences=True) )
            model.add( Dropout(0.2) )
            model.add( BatchNormalization() )
            
            for eachLayer in range(layer-2):
                model.add( GRU(s, return_sequences=True) )
                model.add( Dropout(0.2) )
                model.add( BatchNormalization() )
            
            model.add( GRU(s, return_sequences=False) )
            model.add( Dropout(0.2) )
            model.add( BatchNormalization() )
            
            model.add( Dense(32, activation='relu') )
            model.add( Dropout(0.2) )
            model.add( BatchNormalization() )

            model.add( Dense(1) )
            
            # Set model params and training rates
            opt = l

            model.compile(loss="mse",
                optimizer=opt,
                metrics=['mae'])
            
            name = f"GRU-SEQ{SEQUENCE_LENGTH}-Layers{layer}-Size{s}-RMSprop-Regression-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f"logs/{name}")
            
            # Train model
            model.fit(
            trainX, trainY, 
            epochs=5, 
            validation_data=(testX, testY),
            batch_size=32,
            callbacks=[tensorboard],
            )