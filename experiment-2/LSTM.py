import os
import os.path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pickle
import time
import random
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input, Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adadelta
from tensorflow.keras import regularizers

# Set reproducable seed values to compare each experiment based on their outputs and not seed values
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

# Dataset
dataset = 'BTC_180_30'


# Open dataset from pickle file
pickle_in = open(f"{dataset}.pickle", "rb")
data = pickle.load(pickle_in)
pickle_in.close()

train, test, parameters = data
trainX, trainY = train
testX, testY = test

# Model name
model_name = 'LSTM-180-30'

# Hyper params
DROPOUT = 0.5
EPOCHS = 100
BATCH_SIZE = 64
OPTIMIZER = Adadelta(learning_rate=1.0, rho=0.95)
optimizer_str = 'Adadelta: learning_rate=1.0, rho=0.95'
REGULARIZATION = regularizers.l2(0.01)
regularization_str = 'l2: 0.01, output penalty "activity"'

# Dataset data
INPUT_LEN = parameters['input_len']
OUTPUT_LEN = parameters['output_len']
standardization = parameters['standardization']

# Build the model
# Thanks to https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
# for showing how to make a MANY TO MANY model in Keras
model = Sequential()
model.add( LSTM(INPUT_LEN, input_shape=(trainX.shape[1:]), return_sequences=False, activity_regularizer=REGULARIZATION) )
model.add( Dropout(DROPOUT) )
model.add( BatchNormalization() )

model.add( Dense(INPUT_LEN, activation='relu', activity_regularizer=REGULARIZATION) )
model.add( Dropout(DROPOUT) )
model.add( BatchNormalization() )

model.add( Dense(OUTPUT_LEN, activation='linear') )

model.compile(loss="mse",
                optimizer=OPTIMIZER,
                metrics=['mae'])

# Time callback for tracking epoch training times, from: https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Callbacks
terminateOnNan = TerminateOnNaN()
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='min', baseline=None, restore_best_weights=True)
reduceOnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, min_lr=0.001)
modelCheckpoint = ModelCheckpoint(f'checkpoints/{model_name}.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir=f'logs/{model_name}')
time_callback = TimeHistory()

# Train model
history = model.fit(
    trainX, trainY, 
    epochs=1, 
    validation_data=(testX, testY),
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[
        terminateOnNan,
        earlyStopping,
        reduceOnPlateau,
        modelCheckpoint,
        tensorboard,
        time_callback
    ],
)

# Generate chart
# Using this as assistance: 
# https://cmdlinetips.com/2019/10/how-to-make-a-plot-with-two-different-y-axis-in-python-with-matplotlib/

fig,ax=plt.subplots()

# Primary Axis Labels
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss (MSE)")

# Primary axis data
ax.plot( history.history['loss'], color="orange" )
ax.plot( history.history['val_loss'], color="red" )

# Secondary axis data and labels
ax2=ax.twinx()
ax2.plot( history.history['lr'], color="blue" )
ax2.set_ylabel("Learning Rate" )

# Legend
loss_patch = mpatches.Patch(color='orange', label='Train Loss')
val_loss_patch = mpatches.Patch(color='red', label='Val Loss')
lr_patch = mpatches.Patch(color='blue', label='Learning Rate')
plt.legend(handles=[loss_patch, val_loss_patch, lr_patch])
plt.rcParams["legend.fontsize"] = 12

plt.title(model_name, loc='center')

plt.show()

# save the plot as a file
fig.savefig(f'plots/{model_name}.png', format='png', dpi=250, bbox_inches='tight')

# Store model data to csv for analysis
filePath = 'ml-results.csv'

csvColumns = "Name,Val_Loss,Val_Mae,Epochs_Scheduled,Epochs_Ran,Training_Time(Mins),Input_Len,Output_Len,Batch_Size,Optimizer,Regularization,Dropout"
if not os.path.isfile(filePath):
    f = open(filePath, "a")
    f.write(csvColumns)
    f.close()

df = pd.read_csv(filePath)
df = df[csvColumns.split(',')]

score = model.evaluate(testX, testY, verbose=0)

csvRow = {
    'Name': model_name, 'Val_Loss': score[0], 'Val_Mae': score[1],
    'Epochs_Scheduled': EPOCHS, 'Epochs_Ran': len(history.history['loss']),
    'Training_Time(Mins)': sum(time_callback.times)/60, 'Input_Len': INPUT_LEN, 'Output_Len': OUTPUT_LEN,
    'Batch_Size': BATCH_SIZE, 'Optimizer': optimizer_str, 'Regularization': regularization_str,
    'Dropout': DROPOUT
    }

df = df.append(csvRow, ignore_index=True)

df.to_csv(path_or_buf=filePath, index=False)

print('model-results.csv updated')