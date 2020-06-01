from load_data import load_data
from explore_data import explore_data
import visualise_data as vis
import preprocess_data as pr
import simple_models as smp
import complex_models as cmp
from tensorflow import keras as kr

lr_optimizer = kr.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))
verbose = 0
print("The data of stock prices is being loaded...")
data_frame = load_data()
input("Press Enter to continue...")

print("The data of stock prices is being explored...")
explore_data(data_frame, verbose)
input("Press Enter to continue...")

print("The data of stock prices is being preprocessed...")
data_frame = pr.convert_to_datetime(data_frame, verbose)
data_frames = pr.separate_data(data_frame)
pr.explore_years(data_frames, verbose)
combined_datasets = pr.combine_datasets(data_frames)
if verbose == 1:
    print("We lost " + str(
        data_frame.shape[0] - combined_datasets.shape[0]) + " number of rows after separating the datasets. ")
input("Press Enter to continue...")
# set ticker to Apple Inc. Company
ticker = "AAPL"
attribute = 'close'
date_pts = 10
# year = 2000
apple = 1
print("The stock price data of the ticker " + ticker + " is being visualised...")
vis.visualise(ticker, pr.filter_by_company(combined_datasets, ticker), apple, attribute, date_pts)
input("Press Enter to continue...")
print("The stock price data of the ticker " + ticker + " is being analyzed..")
prices = pr.filter_by_company(data_frames[0], ticker)[attribute].to_numpy()
days = pr.filter_by_company(data_frames[0], ticker)['date'].to_numpy()
print("Stock prediction of the ticker " + ticker + " has started via Linear Regression..")
lr_mse, lr_mae = smp.linear_regression(prices, days, [lr_optimizer], n_days=365, threshold=0.8, window_size=5, batch_size=32,
                                       window_shift=1,
                                       nof_epochs=100, lr_rate=1e-6)
lr = input("Pick the lowest learning rate to train: ")

lr_mse, lr_mae = smp.linear_regression(prices, days, [], n_days=365, threshold=0.8, window_size=5, batch_size=32,
                                       window_shift=1,
                                       nof_epochs=100, lr_rate=float(lr))

print("Stock prediction of the ticker " + ticker + " has started via Deep Neural Networks..")
hidden_neurons = [10]
dnn_mse, dnn_mae = smp.neural_networks(prices, days, hidden_neurons, [lr_optimizer], n_days=365, threshold=0.8,
                                       window_size=5,
                                       batch_size=32,
                                       window_shift=1,
                                       nof_epochs=100, lr_rate=1e-6)
print("Stock prediction of the ticker " + ticker + " has started via Recurrent Neural Networks..")
lr = input("Pick the lowest learning rate to train: ")
dnn_mse, dnn_mae = smp.neural_networks(prices, days, hidden_neurons, [], n_days=365, threshold=0.8,
                                       window_size=5,
                                       batch_size=32,
                                       window_shift=1,
                                       nof_epochs=100, lr_rate=1e-6)


cells = [32, 32]
rnn_mse, rnn_mae = cmp.recurrent_nn(prices, days, cells, [lr_optimizer], n_days=365, threshold=0.8, window_size=5, batch_size=32, window_shift=1,
                                    nof_epochs=100, lr_rate=1e-6)

lr = input("Pick the lowest learning rate to train: ")

rnn_mse, rnn_mae = cmp.recurrent_nn(prices, days, cells, [], n_days=365, threshold=0.8, window_size=5, batch_size=32, window_shift=1,
                                    nof_epochs=100, lr_rate=float(lr))


print("Stock prediction of the ticker " + ticker + " has started via LSTM..")
cells = [32, 32]
lstm_mse, lstm_mae = cmp.lstm(prices, days, cells, [lr_optimizer], n_days=365, bi_directional=False, threshold=0.8, window_size=30,
                              batch_size=32, window_shift=1,
                              nof_epochs=100, lr_rate=1e-6)


lr = input("Pick the lowest learning rate to train: ")
lstm_mse, lstm_mae = cmp.lstm(prices, days, cells, [lr_optimizer], n_days=365, bi_directional=False, threshold=0.8, window_size=30,
                              batch_size=32, window_shift=1,
                              nof_epochs=100, lr_rate=float(lr))






print("Stock prediction of the ticker " + ticker + " has started via Bi-LSTM..")
cells = [32, 32]
bi_lstm_mse, bi_lstm_mae = cmp.lstm(prices, days, cells, bi_directional=True, threshold=0.8, window_size=30,
                                    batch_size=32, window_shift=1,
                                    nof_epochs=100, lr_rate=1e-6)
print("Stock prediction of the ticker " + ticker + " has started via CNN-LSTM..")
cells = [32, 32]
cnn_lstm_mse, cnn_lstm_mae = cmp.cnn_lstm(prices, days, cells, bi_directional=False, threshold=0.8, window_size=30,
                                          batch_size=32, window_shift=1,
                                          nof_epochs=100, lr_rate=1e-6)
print("Stock prediction of the ticker " + ticker + " has started via CNN-bi-LSTM..")
cells = [32, 32]
cnn_bi_lstm_mse, cnn_bi_lstm_mae = cmp.cnn_lstm(prices, days, cells, bi_directional=True, threshold=0.8,
                                                window_size=30,
                                                batch_size=32, window_shift=1,
                                                nof_epochs=100, lr_rate=1e-6)
