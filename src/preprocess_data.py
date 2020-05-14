import pandas as pd


def convert_to_datetime(data):
    print("The date attribute is " + type(data['date'].iloc[0]) +" type object.")
    # The date attribute is string type object.It is better to convert it to datetime object to wrangle.
    data['date'] = pd.to_datetime(data['date'])
    print("The date attribute is " + type(data['date'].iloc[0]) +" type object after transformation.")
    # Now it's date time object.

    print(data['date'].head())
    print("The minimum year of the dataset is: " + str(min(data['date'].dt.year)))
    print("The maximum year of the dataset is: " + str(max(data['date'].dt.year)))
    return data

def separate_data(data):
    # Separating the data according to the economic recessions in the USA.
    # Information about the recession dates retrieved from the site below:
    # https://en.wikipedia.org/wiki/List_of_recessions_in_the_United_States

    after_2009 = data[data['date'].dt.year > 2009]
    between_2001_07 = data[(data['date'].dt.year < 2007) & (data['date'].dt.year > 2001)]
    between_91_2001 = data[(data['date'].dt.year < 2001) & (data['date'].dt.year > 1991)]
    between_82_90 = data[(data['date'].dt.year < 1990) & (data['date'].dt.year > 1982)]
    before_80 = data[data['date'].dt.year < 1980]

    after_2009.head()
    print("The minimum year of the dataset is: " + str(min(after_2009['date'].dt.year)))
    # So, splitting the data works as what we expected.

    print('Row count of After 2009: ', after_2009.shape[0])
    print('Row count of Between 2001 and 2007: ', between_2001_07.shape[0])
    print('Row count of Between 1991 and 2001: ', between_91_2001.shape[0])
    print('Row count of Between 1982 and 1990: ', between_82_90.shape[0])
    print('Row count of Before 1980: ', before_80.shape[0])

    nonrecessed_count = after_2009.shape[0] + between_2001_07.shape[0] + between_91_2001.shape[0] + between_82_90.shape[
        0] + before_80.shape[0]
    print("We lost " + str(data.shape[0] - nonrecessed_count) + " number of rows after separating the datasets. ")

    return data
