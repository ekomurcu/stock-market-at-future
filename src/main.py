from load_data import load_data
from explore_data import explore_data
from visualise_data import visualise
import preprocess_data as pr

print("The data of stock prices is being loaded...")
data_frame = load_data()

print("The data of stock prices is being explored...")
explore_data(data_frame)

print("The data of stock prices is being preprocessed...")
data_frame = pr.convert_to_datetime(data_frame)
data_frames = pr.separate_data(data_frame)
pr.explore_years(data_frames)
combined_datasets = pr.combine_datasets(data_frames)
print("We lost " + str(
    data_frame.shape[0] - combined_datasets.shape[0]) + " number of rows after separating the datasets. ")

# set ticker to Apple Inc. Company
ticker = "AAPL"
attribute = 'high'
date_pts = 10
apple = 1
print("The stock price data of the ticker " + ticker + " is being visualised...")
# visualise(data_frame, ticker)
visualise(ticker, pr.filter_by_company(combined_datasets, ticker), apple, attribute, date_pts)

print("The stock price data of the ticker " + ticker + " is being analyzed..")
print("Stock prediction of the ticker " + ticker + " has started via Linear Regression..")
print("The actual vs. predicted stock price data of the ticker " + ticker + " is being visualised...")
visualise(ticker, pr.filter_by_company(combined_datasets, ticker), apple, attribute, date_pts)
