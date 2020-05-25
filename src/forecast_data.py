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
    mapped_windows = windows_list.map(lambda window: (window[:-1], window[-1:]))
    shuffled_mapped_windows = mapped_windows.shuffle(buffer_size=df.shape[0])
    batched_windows = shuffled_mapped_windows.batch(batch_size).prefetch(1)
    return shuffled_mapped_windows, batched_windows


def print_windows(windows):
    for x, y in windows:
        print("x: ", x.numpy())
        print("y: ", y.numpy())


def compute_predicted(model, series, split_time, window_size):
    predicted_all = []
    for t in range(len(series) - window_size):
        predicted_all.append(model.predict(series[t:t + window_size][np.newaxis]))

    predicted_valid = predicted_all[split_time - window_size:]
    predicted = np.array(predicted_valid)[:, 0, 0]
    return predicted


def evaluate_model(history_model, actual, predicted, test_days):
    vis.plot_mse(history_model)
    mse = kr.metrics.mean_squared_error(actual, predicted).numpy()
    mae = kr.metrics.mean_absolute_error(actual, predicted).numpy()
    print("The MSE of the model is: ", mse)
    print("The MAE of the model is: ", mae)
    print("The actual vs. predicted stock price data of the ticker is being visualised...")
    vis.plot_two(test_days, actual, predicted)
    input("Press Enter to continue...")

    return mse, mae
