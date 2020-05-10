# Read Me

# Experiment 2 Overview
The purpose of this experiment was to see how the five popular RNN architectures would perform predicting regressions on a sequence instead of a single output value.

In this experiment each input to the network was 180 minutes of: Opening price, closing price, volume, high price, and low price data from bitcoin. The outputs were each of the next 30 minutes of closing price data. The goal was to see how well each RNN type could predict each of the next 30 minutes using regressions.

# Experiment issues
As noted below in the "To Run the Experiments" section. The data-generator built for experiment 2 is quite slow as I didn't know about the benefits vectorization gave pandas. Because of this, data-generator.py in experiment 2 is a very slow program to run. Normally I would've substituted this program for the superior one in experiment-3 but I felt that since it was part of the experiment at the time I would leave it in the project.

# To Run the Experiments
The files in experiment 2 rely on a pre-built dataset in the form of a pickle file generated by *experiment-2/data-generator.py*.

The data-generator if ran unchanged will generate the file: *BTC_180_30.pickle*. This file is generated from the 70,000 row file: *data/BTC-USD.csv*

To run any of these experiments do the following (from the main project folder):
1. Generate the dataset 
> python3 experiment-2/data-generator.py
2. Run your choice of experiment
> python3 experiment-2/filename.py

NOTE: the data-generator in experiment 2 is **very slow**. This issue was fixed in experiment 3 but expect the generator to take around 15 minutes to generate the pickle file. Alternatively, you could use the experiment 3 data-generator as it runs in about 15 seconds and performs the same function as this one.

#### If you wish to use the experiment 3 data-generator for faster generation do the following
1. Open experiment-3/data-generator.py
2. Change line 14 to: *FILE_TO_PREDICT = "BTC-USD"*
3. Change line 15 to: *OUTPUT_FILE_NAME = "BTC_180_30"*
4. Run the generator via:
> python3 experiment-3/data-generator

# Logs
Each file is logged using tensorboard to the experiment-2/logs folder. 
Take a look at TensorBoard's documentation for help viewing them: https://www.tensorflow.org/tensorboard

# Checkpoints
The best model for each architecture is saved in the experiment-2/checkpoints folder.
Take a look at this documentation to see how to load the models: https://keras.io/getting_started/faq/

# Plots
A pyplot .png graph of each architecture's training process is saved in the experiment-2/plots folder.

# ml-results.csv
This file tracks a lot of stats about each network. It's stored at: experiment-2/ml-results.csv.