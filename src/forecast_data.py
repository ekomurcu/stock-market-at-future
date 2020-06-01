import tensorflow as tf
from tensorflow import keras as kr
import visualise_data as vis
import numpy as np


def split_data(prices, days, threshold=0.67):
    split = int(len(days) * threshold)
    train_prices = prices[:split]
    train_days = days[:split]
    test_prices = prices[split:]
    test_days = days[split:]
    return split, train_prices, train_days, test_prices, test_days


def create_windows(df, window_size, batch_size, w_shift=1):
    tf_apple = tf.data.Dataset.from_tensor_slices(df)
    windowed_apple = tf_apple.window(window_size + 1, shift=w_shift, drop_remainder=True)
    windows_list = windowed_apple.flat_map(lambda window: window.batch(window_size + 1))
    shuffled_windows = windows_list.shuffle(buffer_size=df.shape[0])
    mapped_windows = shuffled_windows.map(lambda window: (window[:-1], window[-1:]))
    batched_windows = mapped_windows.batch(batch_size).prefetch(1)
    return mapped_windows, batched_windows


def print_windows(windows):
    for x, y in windows:
        print("x: ", x.numpy())
        print("y: ", y.numpy())


def point_prediction(model, series, split_time, window_size, window_shift=1):
    predicted_test = []
    for t in range(split_time - window_size, len(series) - window_size, window_shift):
        predicted_test.append(model.predict(series[t:t + window_size][np.newaxis]))

    predicted_test = np.array(predicted_test)[:, 0, 0]
    return predicted_test


def predict_next_n_days(n, model, series, split_time, window_size):
    predicted_test = []
    window = series[split_time - window_size:split_time]
    if n > len(series[split_time:]):
        print("Try with n values lower than ", len(series[split_time:]) - 1)
    else:
        for i in range(n):
            predicted = model.predict(window[np.newaxis])[0][0]
            predicted_test.append(predicted)
            window = np.delete(window, [0])
            window = np.append(window, predicted)

    # predicted_test=np.array(predicted_test)[:, 0, 0]
    return predicted_test


def model_forecast(model, series,  window_size, batch_size, w_shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=w_shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def evaluate_model(history_model, actual, predicted, test_days, callback):
    if len(callback) == 0:
        vis.plot_mse(history_model)
    else:
        vis.plot_lr(history_model)
    mse = kr.metrics.mean_squared_error(actual, predicted).numpy()
    mae = kr.metrics.mean_absolute_error(actual, predicted).numpy()
    print("The MSE of the model is: ", mse)
    print("The MAE of the model is: ", mae)
    print("The actual vs. predicted stock price data of the ticker is being visualised...")
    vis.plot_two(test_days, actual, predicted)
    input("Press Enter to continue...")

    return mse, mae
