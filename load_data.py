import pandas as pd

FILE_PATH = "C:\\Users\\EGEMEN\\Desktop\\ml_project\\data\\historical_stock_prices.csv"


def load_data():
    # Loading data
    data = pd.read_csv(FILE_PATH)
    return data
