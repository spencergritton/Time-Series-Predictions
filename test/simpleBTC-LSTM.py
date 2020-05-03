import pandas as pd
from datetime import datetime
import os
import numpy as np
from collections import deque
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard

# This model will simply guess if the next price will be higher or lower than the current price
SEQUENCE_LENGTH = 30
FUTURE_DAY_TO_PREDICT = 1
FILE_TO_PREDICT = "BTC"

TRAIN_PERCENT = .80

# Import and combine data into one dataframe
def createMainDf(file):
    dataset = f"data/{file}.csv"
    df = pd.read_csv(dataset)

    df['Date'] = df['Date'].map(lambda a: convertDate(a))   # Convert Date to Linux Epoch Time
    df.set_index("Date", inplace=True)
    df = df[["Close", "Volume"]]
    df = df.astype({f"Volume": float})
    df = df.astype({f"Close": float})

    # Sort by Date
    df.sort_index(inplace=True, ascending=True)

    return cleanData(df)

# Convert String Date to Linux Epoch Time
def convertDate(date):
    timestamp = datetime.strptime(date, '%Y-%m-%d %I-%p')
    return (timestamp-datetime(1970,1,1)).total_seconds()

def cleanData(df):
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep].astype(np.float64)
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)
    return df

# Classify data as 0 if less than or = existing data or 1 if greater
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def addTargets(df):
    df['Future'] = df[f"Close"].shift(-FUTURE_DAY_TO_PREDICT)
    df['Target'] = list(map(classify, df[f"Close"], df["Future"]))
    df = df.drop('Future', 1)
    return cleanData(df)

def splitTrainAndTest(df):
    dateValues = df.index.values
    last_x_pct = dateValues[-int((1-TRAIN_PERCENT)*len(dateValues))]
    
    train = df[df.index < last_x_pct]
    test = df[df.index >= last_x_pct]
    return train, test

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    result.fillna(0, inplace=True)
    return result

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

    # Balance Data (Price going up, Price going down)
    increasing = []
    decreasing = []

    for seq, target in sequential_data:
        if target == 1:
            increasing.append([seq, target])
        elif target == 0:
            decreasing.append([seq, target])
    
    random.shuffle(increasing)
    random.shuffle(decreasing)

    minimumLength = min(len(increasing), len(decreasing))
    increasing = increasing[:minimumLength]
    decreasing = decreasing[:minimumLength]

    sequential_data = increasing + decreasing
    random.shuffle(sequential_data)

    # Create inputs and outputs
    X = []  # inputs
    y = []  # outputs

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.asarray(y)


df = createMainDf(FILE_TO_PREDICT)
df = addTargets(df)
trainDf, testDf = splitTrainAndTest(df)

trainX, trainY = getXandY(trainDf)
testX, testY = getXandY(testDf)

# Build Model
model = Sequential()
model.add( LSTM(128, input_shape=(trainX.shape[1:]), activation='relu', return_sequences=True) )
# model.add( CuDNNLSTM(128, input_shape=(xTrain.shape[1:]), return_sequences=True) )    If running with GPU
model.add( Dropout(0.2) )
model.add( BatchNormalization() )

model.add( LSTM(128, activation='relu', return_sequences=False) )
# model.add( CuDNNLSTM(128) )   If running with GPU
model.add( Dropout(0.1) )
model.add( BatchNormalization() )

model.add( Dense(32, activation='relu') )
model.add( Dropout(0.2) )
model.add( BatchNormalization() )

model.add( Dense(32, activation='relu') )

model.add( Dense(2, activation='softmax') )

# Hyper params
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
    optimizer=opt,
    metrics=['accuracy'])

model.fit(trainX, trainY, epochs=5, validation_data=(testX, testY))

