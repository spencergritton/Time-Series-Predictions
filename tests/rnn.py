import pandas as pd
import os
from datetime import datetime
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

# Prediction parameters
SEQUENCE_LENGTH = 300
PERIOD_PREDICTION = 1
FILE_TO_PREDICT = "BTC"

# Import and combine data into one dataframe
def createMainDf():
    DATA_FILES = [
        "BTC", "LTC", "ETH"
    ]

    main_df = pd.DataFrame()
    for file in DATA_FILES:
        dataset = f"data/{file}.csv"

        df = pd.read_csv(dataset)
        df.rename(columns={"Close": f"{file}_Close", "Pct_Volume": f"{file}_Volume"}, inplace=True)
        df[f"{file}_Volume"] = df[f"{file}_Volume"]
        df['Date'] = df['Date'].map(lambda a: convertDate(a))   # Convert Date to Linux Epoch Time
        df.set_index("Date", inplace=True)
        df = df[[f"{file}_Close", f"{file}_Volume"]]
        df = df.astype({f"{file}_Volume": float})

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)

    return main_df

# Classify data as -1, 0, or 1
def classify(current, future):
    if float(future) > float(current):
        return 1
    elif float(future) == float(current):
        return -1
    else:
        return 0

# Convert String Date to Linux Epoch Time
def convertDate(date):
    timestamp = datetime.strptime(date, '%Y-%m-%d %I-%p')
    return (timestamp-datetime(1970,1,1)).total_seconds()

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def processDf(df):
    df = df.drop('Future', 1)

    # Scale data
    for column in df.columns:
        if column != "Target" and "Volume" in column:
            df.dropna(inplace=True)
            df[column] = preprocessing.scale(df[column].values)
        elif column != "Target":
            df[column] = df[column].pct_change()
            df.dropna(inplace=True)
            df[column] = preprocessing.scale(df[column].values)

    df.dropna(inplace=True)

    sequential_data = []
    previous_days = deque(maxlen=SEQUENCE_LENGTH)

    for i in df.values:
        previous_days.append([n for n in i[:-1]])
        if len(previous_days) == SEQUENCE_LENGTH:
            sequential_data.append([np.array(previous_days), i[-1]])

    random.shuffle(sequential_data)

    # Balance Data (Price going up, Price going down)
    increasing = []
    decreasing = []

    for seq, target in sequential_data:
        if target == 1:
            increasing.append([seq, target])
        elif target == 0:
            decreasing.append([seq, target])
        # Remove any -1's (price not moving)
    
    random.shuffle(increasing)
    random.shuffle(decreasing)

    minimumLength = min(len(increasing), len(decreasing))
    increasing = increasing[:minimumLength]
    decreasing = decreasing[:minimumLength]

    sequential_data = increasing + decreasing
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

# Main

# Setting up main_df targets
main_df = createMainDf()

main_df['Future'] = main_df[f"{FILE_TO_PREDICT}_Close"].shift(-PERIOD_PREDICTION)

main_df['Target'] = list(map(classify, main_df[f"{FILE_TO_PREDICT}_Close"], main_df["Future"]))

# Seperate training and test data
times = sorted(main_df.index.values)
last_10_pct = times[-int(0.1*len(times))]

validation_df = main_df[main_df.index >= last_10_pct]
main_df = main_df[main_df.index < last_10_pct]

train_x, train_y = processDf(main_df)
validation_x, validation_y = processDf(validation_df)

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard

# Build Model
model = Sequential()
model.add( LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True) )
model.add( Dropout(0.2) )
model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add( LSTM(128, activation='relu', return_sequences=True) )
model.add( Dropout(0.2) )
model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add( LSTM(128, activation='relu', return_sequences=False) )
model.add( Dropout(0.2) )
model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add( Dense(32, activation='relu') )
model.add( Dropout(0.2) )

model.add( Dense(2, activation='softmax') )

# Hyper params
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss="sparse_categorical_crossentropy",
    optimizer=opt,
    metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='neuralnets/logs/{}'.format("RNN-3x-128"))
model.fit(train_x, train_y, epochs=30, validation_data=(validation_x, validation_y), callbacks=[tensorboard])