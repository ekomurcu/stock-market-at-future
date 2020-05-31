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


# Plot time series data across days
def plot_series(days, prices, format="-", start=0, end=None):
    plt.plot(days[start:end], prices[start:end], format)
    plt.xlabel("Date")
    plt.ylabel("Prices")
    plt.grid(True)


# Plot the actual  and the predicted test data
def plot_two(test_days, actual, predicted):
    plt.figure(figsize=(18, 9))
    plot_series(test_days, actual)
    plot_series(test_days, predicted)
    plt.show()


# Plot mean squared error of a ML model across epochs
def plot_mse(model_history):
    plt.figure(figsize=(9, 6))
    plt.plot(model_history.history["loss"][5:])
    plt.title('Model Loss')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.show()


def plot_lr(model):
    # plot mse across lr
    plt.figure(figsize=(9, 6))
    plt.semilogx(model.history["lr"], model.history["loss"])
    print(2)
    plt.axis([1e-8, 1e-5, 0, 30])
    plt.title('Model Loss')
    plt.ylabel('MSE')
    plt.xlabel('Learning rate')
    plt.show()
