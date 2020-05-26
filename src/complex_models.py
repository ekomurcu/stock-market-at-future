import forecast_data as fr
import tensorflow as tf
from tensorflow import keras as kr
import numpy as np


def init():
    # clear
    kr.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


# 4.Recurrent Neural Networks
def recurrent_nn(prices, days, cells, threshold=0.67, window_size=30, batch_size=64, window_shift=1, nof_epochs=100,
                 lr_rate=1e-6, mom=0.9):
    # Split data
    split, train_prices, train_days, test_prices, test_days = fr.split_data(prices, days, threshold=threshold)
    # Reset all internal variables
    init()
    # Create windows on training data
    train_windows, train_batch = fr.create_windows(train_prices, window_size=window_size, batch_size=batch_size,
                                                   w_shift=window_shift)
    # Introduce model
    rnn_model = kr.models.Sequential()
    rnn_model.add(kr.layers.Lambda(lambda data: tf.expand_dims(data, axis=-1),
                                   input_shape=[None]))  # pre-process the input dim for uni-variate analysis
    for cell in cells[:-1]:
        rnn_model.add(kr.layers.SimpleRNN(units=cell, return_sequences=True))  # hidden layer with relu activation

    rnn_model.add(kr.layers.SimpleRNN(units=cells[-1], return_sequences=False))  # hidden layer with relu activation
    rnn_model.add(kr.layers.Dense(1))  # output layer
    rnn_model.add(kr.layers.Lambda(lambda x: 100.0 * x))  # scaling
    rnn_model.summary()
    # Choose Huber loss function which is less susceptible to outliers and noise.
    rnn_model.compile(loss=kr.losses.Huber(), optimizer=kr.optimizers.SGD(lr=lr_rate, momentum=mom),
                      metrics=["mae", "mse"])
    history_rnn = rnn_model.fit(train_batch, epochs=100)
    # Predictions
    predicted_rnn = fr.compute_predicted(rnn_model, prices, split, window_size)
    mse, mae = fr.evaluate_model(history_rnn, test_prices, predicted_rnn, test_days)
    return mse, mae


# cell types
# forward
def uni_directional_layers(lstm_model, cells):
    for cell in cells[:-1]:
        lstm_model.add(
            kr.layers.LSTM(units=cell, return_sequences=True))  # hidden layer with relu activation

    lstm_model.add(
        kr.layers.LSTM(units=cells[-1], return_sequences=False))  # hidden layer with relu activation
    return lstm_model


# forward and backward
def bi_directional_layers(lstm_model, cells):
    for cell in cells[:-1]:
        lstm_model.add(
            kr.layers.Bidirectional(
                kr.layers.LSTM(units=cell, return_sequences=True)))  # hidden layer with relu activation

    lstm_model.add(
        kr.layers.Bidirectional(
            kr.layers.LSTM(units=cells[-1], return_sequences=False)))  # hidden layer with relu activation
    return lstm_model


# 5. LSTM
def lstm(prices, days, cells, bi_directional=True, threshold=0.67, window_size=30, batch_size=64, window_shift=1,
         nof_epochs=100,
         lr_rate=1e-6, mom=0.9):
    # Split data
    split, train_prices, train_days, test_prices, test_days = fr.split_data(prices, days, threshold=threshold)
    # Reset all internal variables
    init()
    # Create windows on training data
    train_windows, train_batch = fr.create_windows(train_prices, window_size=window_size, batch_size=batch_size,
                                                   w_shift=window_shift)
    # Introduce model
    lstm_model = kr.models.Sequential()
    lstm_model.add(kr.layers.Lambda(lambda data: tf.expand_dims(data, axis=-1),
                                    input_shape=[None]))  # pre-process the input dim for uni-variate analysis
    if bi_directional:
        lstm_model = bi_directional_layers(lstm_model, cells)
    else:
        lstm_model = uni_directional_layers(lstm_model, cells)
    lstm_model.add(kr.layers.Dense(1))  # output layer
    lstm_model.add(kr.layers.Lambda(lambda x: 100.0 * x))  # scaling
    lstm_model.summary()

    # Choose Huber loss function which is less susceptible to outliers and noise.
    lstm_model.compile(loss=kr.losses.Huber(), optimizer=kr.optimizers.SGD(lr=lr_rate, momentum=mom),
                       metrics=["mae", "mse"])
    history_lstm = lstm_model.fit(train_batch, epochs=100)
    # Predictions
    predicted_lstm = fr.compute_predicted(lstm_model, prices, split, window_size)
    mse, mae = fr.evaluate_model(history_lstm, test_prices, predicted_lstm, test_days)
    return mse, mae
