from load_data import load_data
from explore_data import explore_data
import visualise_data as vis
import preprocess_data as pr
import simple_models as simple

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
attribute = 'high'
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
lr_mse, lr_mae = simple.linear_regression(prices, days, threshold=0.67, window_size=30, batch_size=32, window_shift=1,
                                          nof_epochs=100, lr_rate=1e-6)
print("Stock prediction of the ticker " + ticker + " has started via Neural Networks..")
hidden_neurons = [10]
lr_mse, lr_mae = simple.neural_networks(prices, days, hidden_neurons, threshold=0.67, window_size=30, batch_size=32,
                                        window_shift=1,
                                        nof_epochs=100, lr_rate=1e-6)
