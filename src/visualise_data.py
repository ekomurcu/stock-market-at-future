import matplotlib.pyplot as plt


# Visualise predicted and actual stock prices
# Send the multi argument as True if multi time-series are to be displayed.
def visualise(ticker, actual, predicted, attribute, nof_datepts, multi=False):
    plt.figure(figsize=(18, 9))
    plt.grid()
    plt.plot(range(actual.shape[0]), actual[attribute])
    if multi:
        plt.plot(range(len(predicted)), predicted)
    plt.xticks(range(0, actual.shape[0], int(actual.shape[0] / nof_datepts) + 1),
               actual['date'].loc[::int(actual.shape[0] / nof_datepts) + 1], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(str(attribute) + ' Prices of ' + str(ticker), fontsize=18)
    plt.show()
