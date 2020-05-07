from load_data import load_data
from explore_data import explore_data
from preprocess_data import convert_to_datetime, separate_data
from visualise_data import visualise

data_frame = load_data()
explore_data(data_frame)
data_frame = convert_to_datetime(data_frame)
data_frame = separate_data(data_frame)
# set ticker to Apple Inc. Company
ticker = "AAPL"
visualise(data_frame, ticker)
