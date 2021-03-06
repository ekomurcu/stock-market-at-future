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


# 1.MVA
def moving_average():
    return 1


# 2.Linear Regression
def linear_regression(prices, days, callback, n_days, threshold=0.67, window_size=30, batch_size=64, window_shift=1,
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
    lr_layer = kr.layers.Dense(1, input_shape=[window_size])  # single layer with one neuron
    lr_model = kr.models.Sequential(lr_layer)
    lr_model.summary()
    # Choose mse loss function and sgd optimizer
    lr_model.compile(loss="mse", optimizer=kr.optimizers.SGD(lr=lr_rate, momentum=mom))
    if len(callback) == 0:
        history_lr = lr_model.fit(train_batch, epochs=4 * nof_epochs)
        print("Parameters")
        print(lr_layer.get_weights())
        # Predictions
        print("Point by point prediction")
        predicted_lr = fr.point_prediction(lr_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_lr, test_prices, predicted_lr, test_days, callback)
        print("Predict next n days function")
        predicted_lr = fr.predict_next_n_days(n_days, lr_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_lr, test_prices[:n_days], predicted_lr, test_days[:n_days], callback)

    else:
        history_lr = lr_model.fit(train_batch, callbacks=callback, epochs=nof_epochs)
        vis.plot_lr(lr_model)
        mse, mae = 0, 0
    return mse, mae


# 3.Neural Networks
def neural_networks(prices, days, hidden_neurons, callback, n_days, threshold=0.67, window_size=30, batch_size=64,
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
    dnn_model = kr.Sequential()
    dnn_model.add(
        kr.layers.Dense(units=10, input_shape=[window_size], activation='relu'))  # input layer with relu activation
    for neuron in hidden_neurons:
        dnn_model.add(kr.layers.Dense(units=neuron, activation='relu'))  # hidden layer with relu activation
        # dnn_model.add(kr.layers.Dropout(0.1)) # add dropout to prevent over-fitting
    dnn_model.add(kr.layers.Dense(1))  # output layer
    dnn_model.summary()
    # Choose mse loss function and sgd optimizer
    dnn_model.compile(loss="mse", optimizer=kr.optimizers.SGD(lr=lr_rate, momentum=mom))
    if len(callback) == 0:
        history_dnn = dnn_model.fit(train_batch, epochs=4 * nof_epochs)
        # Predictions
        print("Point by point prediction")
        predicted_dnn = fr.point_prediction(dnn_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_dnn, test_prices, predicted_dnn, test_days, callback)
        print("Predict next n days function")
        predicted_dnn = fr.predict_next_n_days(n_days, dnn_model, prices, split, window_size)
        mse, mae = fr.evaluate_model(history_dnn, test_prices[:n_days], predicted_dnn, test_days[:n_days], callback)

    else:
        history_dnn = dnn_model.fit(train_batch, callbacks=callback, epochs=nof_epochs)
        vis.plot_lr(history_dnn)
        mse, mae = 0, 0
    return mse, mae
