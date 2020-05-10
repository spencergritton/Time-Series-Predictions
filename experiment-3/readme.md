# Read Me

# Experiment 3 Overview
The purpose of this experiment is to see what the maximum potential of RNNs may be on this dataset. This was accomplished using a few alterations to experiment 2:
- Multiple RNN layers
- Multiple Dense layers
- 250 epoch training (~21 hours)
- 1,000,000 minutes of total data (~15X more than experiment 2)

# Experiment issues
There were several large issues with this experiment
- Unfortunately due to the price of renting a GPU and training times, I was only able to train one network.
- Attempting to use the experiment 2 data-generator.py on a 1,000,000 row csv file was **extremely** slow. The program ran for 27 hours on my mac and then crashed when saving the dataset pickle file. Because of this I was forced to learn more about pandas and pickle to solve this issue.
- Loading the dataset into memory was impossible on many machines. This was solved by renting a computer with more memory.

# To Run the Experiments
The files in experiment-3 rely on a pre-built dataset in the form of a pickle file generated by *experiment-3/data-generator.py*.

The data-generator if ran unchanged will generate the file: *coinbase.pickle*. This file is generated from the 1,000,000 row file: *data/coinbase.csv*

So to run any of these experiments do the following (from the main project folder):
1. Generate the dataset 
> python3 experiment-3/data-generator.py
2. Run your choice of experiment
> python3 experiment-3/filename.py

**NOTE: The dataset for this experiment is around 10GB and will take about 1-5 minutes to generate depending on CPU.**
**NOTE 2: Loading the dataset into python memory requires > 10GB of RAM. If your computer does not have this the program will crash. Additionally, if your GPU has low memory you may have to reduce the batch size.**

# Logs
Each file is logged using tensorboard to the experiment-2/logs folder. 
Take a look at TensorBoard's documentation for help viewing them: https://www.tensorflow.org/tensorboard

# Checkpoints
The best Keras for each architecture is saved in the experiment-3/checkpoints folder.
Take a look at this documentation to see how to load the models: https://keras.io/getting_started/faq/

# Plots
A pyplot .png graph of each architecture's training process is saved in the experiment-3/plots folder. Check them out if you're curious.

# ml-results.csv
This file tracks a lot of stats about each network. It's stored at: experiment-3/ml-results.csv.