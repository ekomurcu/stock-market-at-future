import forecast_data as fr
import tensorflow as tf
from tensorflow import keras as kr
import numpy as np
import visualise_data as vis


def init():
    # clear
    kr.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


# 4.Recurrent Neural Networks
def recurrent_nn(prices, days, cells, callback, n_days, threshold=0.67, window_size=30, batch_size=64, window_shift=1,
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

    # Predictions
    # Predictions
    if len(callback) == 0:
        history_rnn = rnn_model.fit(train_batch, epochs=4 * nof_epochs)
        # Predictions
        print("Point by point prediction")
        predicted_rnn = fr.point_prediction(rnn_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_rnn, test_prices, predicted_rnn, test_days, callback)
        print("Predict next n days function")
        predicted_rnn = fr.predict_next_n_days(n_days, rnn_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_rnn, test_prices[:n_days], predicted_rnn, test_days[:n_days], callback)

    else:
        history_rnn = rnn_model.fit(train_batch, epochs=nof_epochs, callbacks=callback)
        vis.plot_lr(history_rnn)
        mse, mae = 0, 0

    return mse, mae


# cell types
# forward
def uni_directional_layers(lstm_model, cells, sequence=False):
    for cell in cells[:-1]:
        lstm_model.add(
            kr.layers.LSTM(units=cell, return_sequences=True))  # hidden layer with relu activation
    if sequence:
        lstm_model.add(
            kr.layers.LSTM(units=cells[-1], return_sequences=True))
    elif cells[-1]:
            lstm_model.add(
                kr.layers.LSTM(units=cells[-1], return_sequences=False))  # hidden layer with relu activation
    return lstm_model


# forward and backward
def bi_directional_layers(lstm_model, cells, sequence=False):
    for cell in cells[:-1]:
        lstm_model.add(
            kr.layers.Bidirectional(
                kr.layers.LSTM(units=cell, return_sequences=True)))  # hidden layer with relu activation
    if sequence:
        lstm_model.add(
            kr.layers.LSTM(units=cells[-1], return_sequences=True))
    elif cells[-1]:
        lstm_model.add(
            kr.layers.LSTM(units=cells[-1], return_sequences=False))  # hidden layer with relu activation
    return lstm_model


# 5. LSTM
def lstm(prices, days, cells, callback, n_days, sequence=False, bi_directional=True, threshold=0.67, window_size=30, batch_size=64,
         window_shift=1,
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
    # Predictions
    if len(callback) == 0:
        history_lstm = lstm_model.fit(train_batch, epochs=4 * nof_epochs)
        # Predictions
        print("Point by point prediction")
        predicted_lstm = fr.point_prediction(lstm_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_lstm, test_prices, predicted_lstm, test_days, callback)
        print("Predict next n days function")
        predicted_lstm = fr.predict_next_n_days(n_days, lstm_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_lstm, test_prices[:n_days], predicted_lstm, test_days[:n_days], callback)

    else:
        history_lstm = lstm_model.fit(train_batch, epochs=nof_epochs, callbacks=callback)
        vis.plot_lr(history_lstm)
        mse, mae = 0, 0

    return mse, mae


def cnn_lstm(prices, days, cells, callback, n_days, bi_directional=True, threshold=0.67, window_size=30, batch_size=64,
             window_shift=1,
             nof_epochs=100,
             lr_rate=1e-6, mom=0.9):
    # Split data
    split, train_prices, train_days, test_prices, test_days = fr.split_data(prices, days, threshold=threshold)
    # Reset all internal variables
    init()
    train_prices = tf.expand_dims(train_prices, axis=-1)
    # Create windows on training data
    train_windows, train_batch = fr.create_windows(train_prices, window_size=window_size, batch_size=batch_size,
                                                   w_shift=window_shift)
    # Introduce model
    cnn_lstm_model = kr.models.Sequential()
    cnn_lstm_model.add(kr.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu',
                                        input_shape=[None, 1]))  # pre-process the input dim for uni-variate analysis
    cells.append(0)
    if bi_directional:
        cnn_lstm_model = bi_directional_layers(cnn_lstm_model, cells)
    else:
        cnn_lstm_model = uni_directional_layers(cnn_lstm_model, cells)
    cnn_lstm_model.add(kr.layers.Dense(1))  # output layer
    cnn_lstm_model.add(kr.layers.Lambda(lambda x: 200.0 * x))  # scaling
    cnn_lstm_model.summary()

    # Choose Huber loss function which is less susceptible to outliers and noise.
    cnn_lstm_model.compile(loss=kr.losses.Huber(), optimizer=kr.optimizers.SGD(lr=lr_rate, momentum=mom),
                           metrics=["mae", "mse"])
    # Predictions
    if len(callback) == 0:
        history_cnn_lstm = cnn_lstm_model.fit(train_batch, epochs=4 * nof_epochs)
        # Predictions
        # print("Point by point prediction")
        # predicted_cnn_lstm = fr.point_prediction(cnn_lstm_model, prices, split, window_size)
        # mse, mae = fr.evaluate_model(history_cnn_lstm, test_prices, predicted_cnn_lstm, test_days, callback)
        # print("Predict next n days function")
        # predicted_cnn_lstm = fr.predict_next_n_days(n_days, cnn_lstm_model, prices, split, window_size)
        # mse, mae = fr.evaluate_model(history_cnn_lstm, test_prices[:n_days], predicted_cnn_lstm, test_days[:n_days], callback)
        print("Model forecast function")
        predicted_cnn_lstm = fr.model_forecast(cnn_lstm_model, prices[..., np.newaxis], window_size, batch_size)
        predicted_cnn_lstm = predicted_cnn_lstm[split_time - window_size:-1, -1, 0]
        mse, mae = fr.evaluate_model(history_cnn_lstm, test_prices, predicted_cnn_lstm, test_days, callback)

    else:
        history_cnn_lstm = cnn_lstm_model.fit(train_batch, epochs=nof_epochs, callbacks=callback)
        mse, mae = 0, 0

    return mse, mae
