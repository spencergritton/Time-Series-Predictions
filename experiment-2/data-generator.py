import pandas as pd
import os
import numpy as np
from collections import deque
import random
import pickle
import time

# Dataset parameters
SEQUENCE_LENGTH = 180
NUMBER_MINUTES_TO_PREDICT = 30
FILE_TO_PREDICT = "BTC-USD"
outputFileName = 'BTC_180_30'
TRAIN_PERCENT = .80

# Import data into one df with columns: Volume, Close, Low, High, Open, Date
def createMainDf(file):
    dataset = f"data/{file}.csv"
    df = pd.read_csv(dataset)

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    df = cleanData(df)

    df = df.astype({f"Volume": float})
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

# Create input sequences using vectors for pandas
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
            df['Volume'].values.tolist(),
        ]
    ] * len(df['Index'].values.tolist())

    # Reduce the input sequences to the correct values
    df['InputSequence'] = df.apply(reduceSequences, axis=1)
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
trainDf = addTargets(trainDf)
print('Added target sequences')

testDf = addTargets(testDf)
print('Added target sequences')

# Add Input Sequences
trainDf = addInputSequences(trainDf)
print('Added input sequences')

testDf = addInputSequences(testDf)
print('Added input sequences')

# Get training and testing data in sequences of appropriate form
trainX = np.array( trainDf['InputSequence'].values.tolist() )
trainY = np.array( trainDf['Target'].values.tolist() )
testX = np.array( testDf['InputSequence'].values.tolist() )
testY = np.array( testDf['Target'].values.tolist() )
print('Converted data to Keras readable format')

# Save training and testing data to the following format
# train = (X, Y)
# test = (X, Y)
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

# Save to pickle file
pickle_stream = open(f"{outputFileName}.pickle", "wb")
pickle.dump(data, pickle_stream)
pickle_stream.close()

print(f'Pickle saved: {outputFileName}')
print(f"Dataset generator took {time.time()-start_time} seconds to generate the dataset.")