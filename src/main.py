from load_data import load_data
from explore_data import explore_data
from preprocess_data import convert_to_datetime, separate_data
from visualise_data import visualise

print("The data of stock prices is being loaded...")
data_frame = load_data()

print("The data of stock prices is being explored...")
explore_data(data_frame)

print("The data of stock prices is being preprocessed...")
data_frame = convert_to_datetime(data_frame)
data_frame = separate_data(data_frame)

# set ticker to Apple Inc. Company
ticker = "AAPL"
print("The stock price data of the ticker " + ticker + " is being visualised...")
visualise(data_frame, ticker)

print("The stock price data of the ticker " + ticker + " is being analyzed..")
print("Stock prediction of the ticker " + ticker + " has started via Linear Regression..")
