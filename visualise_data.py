import matplotlib.pyplot as plt


# Visualise the timeline of the mid price of the given ticker from the given dataframe.
def visualise(data, ticker):
    after_2009 = data[data['date'].dt.year > 2009]
    after_2009.head()
    ticker_df = after_2009[after_2009['ticker'] == ticker]
    ticker_df.head()
    print("There are " + str(after_2009[after_2009['ticker'] == ticker].shape[0]) + " stock information about " + str(
        ticker))
    # Visualise stock prices with 50 intervals
    plt.figure(figsize=(18, 9))
    plt.grid()
    plt.plot(range(ticker_df.shape[0]), (ticker_df['low'] + ticker_df['high']) / 2.0)
    plt.xticks(range(0, ticker_df.shape[0], 50), ticker_df['date'].loc[::50], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price of ' + str(ticker), fontsize=18)
    plt.show()
