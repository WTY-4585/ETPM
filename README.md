Code for **Dynamic tactile process sensing on the basis of fingerprint-like sensor array and early temporal perception model**

### 1. ETPM.py

**Overview**<br />
This repository contains code for training an Early Classification model architecture with the purpose of making early predictions in time-series classification tasks. The model is designed to predict the class label of a given time-series sequence at an early time step while maximizing classification performance and minimizing the time taken for prediction.<br />

**Dependencies**<br />
Python 3.x, PyTorch, Scikit-learn, NumPy, Pandas, tqdm, argparse

**Methods and Classes**<br />
- parse_args()<br />
This function parses command-line arguments to configure the training process. The parameters include:
    - alpha: Coefficient for the early reward loss function.
    - epsilon: Threshold for the early stopping reward function.
    - learning-rate: Learning rate for the optimizer.
    - weight-decay: Weight decay for the optimizer.
    - epochs: Number of training epochs.
    - sequencelength: Length of each input sequence.
    - batchsize: Batch size for training.

- EarlyNN()<br />
This is the CNN-LSTM model used for early classification. It is designed to process sequential data and make predictions as early as possible within the sequence. The architecture consists of:
    - Input layer with dimensionality input_dim.
    - One or more CNN-LSTM layers, each with hidden_dim number of hidden units.
    - A stopping decision head that determines when to make the final prediction.
    - Dropout layers to prevent overfitting.
 
- EarlyRewardLoss<br />
A custom loss function that combines traditional classification loss with an additional reward term based on how early the prediction is made.
The reward encourages the model to make predictions sooner rather than later, helping to reduce prediction latency.
    - alpha: A hyperparameter that controls the trade-off between classification loss and early reward. 
    - weight: Used to provide class weights for the negative log-likelihood loss (NLLLoss) calculation.

 ****
 
 ### 2. CNNLSTM.py
 **Overview**<br />
 This Python file implements a Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) network for time-series classification.<br />

 **Dependencies**<br />
 Python 3.x, PyTorch, NumPy

 ****

 ### 3. Data
  **Overview**<br />
 A small dataset used to demonstrate the model's performance.<br />
