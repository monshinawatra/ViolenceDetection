from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM


def get_model():
    chunk_size = 4096
    n_chunks = 20
    rnn_size = 512

    model = Sequential()
    model.add(LSTM(rnn_size, input_shape=(n_chunks, chunk_size)))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model