import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import string
import keras


# DONE: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    y = series[window_size:]
    series_size = len(series) - window_size
    i = 0
    while i < series_size:
        X.append(series[i: i + window_size])
        i = i + 1

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model

### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    text = text.lower()
    text_unique_characters = ''.join(set(text))
    for c in text_unique_characters:
        if c != ' ' and c not in string.ascii_lowercase and c not in punctuation:
            text = text.replace(c, '')

    return text

### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    number_of_pairs = -(-len(text) // step_size)
    i = 0
    while i < number_of_pairs:
        start_index = i * step_size
        inputs.append(text[start_index: start_index + window_size])
        outputs.append(text[start_index + window_size: start_index + window_size + 1])
        i = i + 1

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
