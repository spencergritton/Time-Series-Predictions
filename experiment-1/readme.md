# Read Me

### Experiment 1 Overview
The purpose of this experiment was to test if there was a clearly superior RNN neural network architecture
in terms of convergence speed and accuracy for financial time series regressions.

After this experiment I realized that there were many issues with the experiment including:
- Not properly normalizing the data
- Not shuffling the data properly
- Training for a low number of epochs
- Not recording enough data
- Not using regularization
- Not change learning rate at all throughout training
- Not use any keras callbacks to react to how model was training

Normally I would've removed the experiment from the project but I felt that it was a good first step
to learning about creating a time series regression network and an important part of the learning process,
so I left it in.

# To Run the Experiments
To run any of these files simply navigate to the main project directory and type the following:
> python3 experiment-1/filename.py

Each file will run on its own with no other dependencies (this will not be the case in the next experiments).

# Logs
Each file is logged using tensorboard to the experiment-1/logs folder. 
Take a look at TensorBoard's documentation for help viewing them: https://www.tensorflow.org/tensorboard