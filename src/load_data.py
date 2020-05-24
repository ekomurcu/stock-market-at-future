import pandas as pd

FILE_PATH = "..\\data\\historical_stock_prices.csv"


def load_data():
    # Loading data
    data = pd.read_csv(FILE_PATH)
    return data
