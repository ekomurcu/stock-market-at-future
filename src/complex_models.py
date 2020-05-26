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
def recurrent_nn(prices, days, threshold=0.67, window_size=30, batch_size=64, window_shift=1, nof_epochs=100,
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
    rnn_model.add(kr.layers.SimpleRNN(units=40, return_sequences=True))  # hidden layer with relu activation
    rnn_model.add(kr.layers.SimpleRNN(units=40, return_sequences=False))  # hidden layer with relu activation
    rnn_model.add(kr.layers.Dense(1))  # output layer
    rnn_model.add(kr.layers.Lambda(lambda x: 100.0 * x))  # scaling
    # Choose Huber loss function which is less susceptible to outliers and noise.
    rnn_model.compile(loss=kr.losses.Huber(), optimizer=kr.optimizers.SGD(lr=lr_rate, momentum=mom),
                      metrics=["mae", "mse"])
    history_rnn = rnn_model.fit(train_batch, epochs=100)
    # Predictions
    predicted_rnn = fr.compute_predicted(rnn_model, prices, split, window_size)
    mse, mae = fr.evaluate_model(history_rnn, test_prices, predicted_rnn, test_days)
    return mse, mae
