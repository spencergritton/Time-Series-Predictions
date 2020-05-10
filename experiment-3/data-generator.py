import pandas as pd
import os
import numpy as np
from collections import deque
import random
import pickle
import time
import gc

# Dataset parameters
# Warning, with current parameters the dataset created will be roughly 10GB
SEQUENCE_LENGTH = 180
NUMBER_MINUTES_TO_PREDICT = 30
FILE_TO_PREDICT = "coinbase"
OUTPUT_FILE_NAME = 'coinbase_180_30'
TRAIN_PERCENT = .80

# Import data into one df with columns: Date, Open, Close, Low, High, Volume_BTC, Volume_Dollars, Weighted_Price
def createMainDf(file):
    dataset = f"data/{file}.csv"
    df = pd.read_csv(dataset)

    df = df[["Date", "Open", "Close", "High", "Low", "Volume_BTC", "Volume_Dollars", "Weighted_Price"]]

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

# Removes null, invalid, NaN data
def cleanData(df):
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep].astype(np.float64)
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)
    return df

# Set the target for each individual row
def setEachTarget(index, l):
    if index + 1 + NUMBER_MINUTES_TO_PREDICT > len(l):
        return "na"
    return l[index + 1 : index + 1 + NUMBER_MINUTES_TO_PREDICT]

# Add target column (Y of dataset)
def addTargets(df):
    # Set temporary index column for setting targets
    df["Index"] = [ *range(0, len( df.index.values) ) ]

    # Set each target column to maximum amount of targets
    closePriceList = df.Close.values.tolist()
    df['Target'] = [setEachTarget(x, closePriceList) for x in df['Index']]
    # Remove invalid rows
    df = df[df.Target != "na"]

    return df

# Split main df into training and testing data based on date
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

    # Normalize each feature
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

    # Normalize each feature with std and mean of training set
    for feature_name in df.columns:
        mean, std = standardization[feature_name]
        result[feature_name] = (df[feature_name] - mean) / std

    result.fillna(mean, inplace=True)

    return result

# Set input sequence for the row called on
def setEachSequence(index, openL, closeL, highL, lowL, btcL, dollarsL, weightedL):
    if index < SEQUENCE_LENGTH:
        return "na"
    return [
        openL[index - SEQUENCE_LENGTH : index],
        closeL[index - SEQUENCE_LENGTH : index],
        highL[index - SEQUENCE_LENGTH : index],
        lowL[index - SEQUENCE_LENGTH : index],
        btcL[index - SEQUENCE_LENGTH : index],
        dollarsL[index - SEQUENCE_LENGTH : index],
        weightedL[index - SEQUENCE_LENGTH : index],
        ]

# Set input sequence column for each row in the df
def addInputSequences(df):
    # Set temporary index column for setting targets
    df["Index"] = [ *range(0, len( df.index.values) ) ]

    # Call setSequence for each row in the df
    ol = df.Open.values.tolist()
    cl = df.Close.values.tolist()
    hl = df.High.values.tolist()
    ll = df.Low.values.tolist()
    vbl = df.Volume_BTC.values.tolist()
    vdl = df.Volume_Dollars.values.tolist()
    wpl = df.Weighted_Price.values.tolist()

    df['InputSequence'] = [setEachSequence(x, ol, cl, hl, ll, vbl, vdl, wpl) for x in df['Index']]

    # Remove invalid rows
    df = df[df.InputSequence != "na"]

    return df


# Main code to execute to generate dataset
start_time = time.time()

df = createMainDf(FILE_TO_PREDICT)
print('Created main df')

# Split into training and testing dataframe
trainDf, testDf = splitTrainAndTest(df)
print('Split data into training and testing')

# Drop date column
trainDf = trainDf.drop('Date', axis=1)
testDf = testDf.drop('Date', axis=1)

# Normalize training, normalize testing based on training normalization
trainDf, standardization = normalizeTrain(trainDf)
testDf = normalizeTest(testDf, standardization)
print('Normalized data')

# Add target sequences
start_time_timer = time.time()
trainDf = addTargets(trainDf)
print(f'Added target sequences: {time.time()-start_time_timer} seconds')

start_time_timer = time.time()
testDf = addTargets(testDf)
print(f'Added target sequences: {time.time()-start_time_timer} seconds')

# Add Input Sequences
start_time_timer = time.time()
trainDf = addInputSequences(trainDf)
print(f'Added input sequences: {time.time()-start_time_timer} seconds')

start_time_timer = time.time()
testDf = addInputSequences(testDf)
print(f'Added input sequences: {time.time()-start_time_timer} seconds')

# Free up allocated memory
gc.collect()

# Get training and testing data in sequences of appropriate form
print('Converting data to Keras format...')
trainX = np.array( trainDf['InputSequence'].values.tolist() )
trainY = np.array( trainDf['Target'].values.tolist() )
testX = np.array( testDf['InputSequence'].values.tolist() )
testY = np.array( testDf['Target'].values.tolist() )
print('Converted data to Keras readable format')

# Save training and testing data to the following format
# train = (trainX, trainY)
# test = (testX, testY)
# parameters = { 
# 'input_len': 300, 'output_len': 60, 
# 'standardization': {'feature1': (mean, std), 'feature2': (mean, std), ...}
# }

# data = (train, test, parameters)

# Create data to be saved
train = trainX, trainY
test = testX, testY
parameters = {'input_len': SEQUENCE_LENGTH, 'output_len': NUMBER_MINUTES_TO_PREDICT, 'standardization': standardization}
data = train, test, parameters

# Free up allocated memory
gc.collect()

# Save to pickle file
print('Saving file...')
pickle.dump(data, open(f"{OUTPUT_FILE_NAME}.pickle", 'wb'), protocol=4)

print(f'Pickle saved: {OUTPUT_FILE_NAME}.pickle')
print(f"Dataset generator took {time.time()-start_time} seconds to generate the dataset.")