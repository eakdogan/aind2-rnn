import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    #loop over the input series
    for point in range(len(series) - window_size):
        #extract the input points by window size
        X.append(series[point:point+window_size])
        #extract the output point by looking at the next point
        y.append(series[point+window_size])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)


    # TODO: build an RNN to perform regression on our time series input/output data
    #model type
    model = Sequential()
    #lstm module 5 nodes
    model.add(LSTM(5, input_shape=(1,window_size)))
    #dense layer for output 1 node
    model.add(Dense(1))
    model.summary()


    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    ### TODO: list all unique characters in the text and remove any non-english ones
    # find all unique characters in the text

    # remove as many non-english characters and character sequences as you can
    #regex removal of non alphanumerics
    text = re.sub('[^a-zA-Z!",.?]', ' ', text)
    # shorten any extra dead space created above
    text = text.replace('  ',' ')

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # containers for input/output pairs
    inputs = []
    outputs = []

    #loop widnow over text with step size
    for step in range(0, len(text) - window_size, step_size):
        #add the text in window
        inputs.append(text[step:step+window_size])
        #add the successor
        outputs.append(text[step+window_size])

    return inputs,outputs
