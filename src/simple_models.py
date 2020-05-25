import forecast_data as fr
import tensorflow as tf
from tensorflow import keras as kr
import numpy as np


def init():
    # clear
    kr.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


# 1.MVA
def moving_average():
    return 1


# 2.Linear Regression
def linear_regression(prices, days, threshold=0.67, window_size=30, batch_size=64, window_shift=1, nof_epochs=100,
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
    history_lr = lr_model.fit(train_batch, epochs=nof_epochs)
    print("Parameters")
    print(lr_layer.get_weights())
    # Predictions
    predicted = fr.compute_predicted(lr_model, prices, split, window_size)
    mse, mae = fr.evaluate_model(history_lr, test_prices, predicted, test_days)
    return mse, mae