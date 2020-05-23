import pandas as pd


def convert_to_datetime(data):
    print("The date attribute is " + str(type(data['date'].iloc[0])) + " type object.")
    # The date attribute is string type object.It is better to convert it to datetime object to wrangle.
    data['date'] = pd.to_datetime(data['date'])
    print("The date attribute is " + str(type(data['date'].iloc[0])) + " type object after transformation.")
    # Now it's date time object.

    print(data['date'].head())
    print("The minimum year of the dataset is: " + str(min(data['date'].dt.year)))
    print("The maximum year of the dataset is: " + str(max(data['date'].dt.year)))
    return data


# Send the parameter as zero to be displayed until the start or end.
def filter_by_year(data, start=0, end=0):
    if start and end:
        return data[(data['date'].dt.year < end) & (data['date'].dt.year > start)]
    elif not start:
        return data[(data['date'].dt.year < end)]
    elif not end:
        return data[(data['date'].dt.year > start)]


def filter_by_company(whole_data, stock):
    return whole_data[whole_data['ticker'] == stock]


def explore_years(data_frames):
    print("The minimum year of the dataset is: " + str(min(data_frames[0]['date'].dt.year)))
    print("So, splitting the data works as what we expected.")
    print('Row count of After 2009: ', data_frames[0].shape[0])
    print('Row count of Between 2001 and 2007: ', data_frames[1].shape[0])
    print('Row count of Between 1991 and 2001: ', data_frames[2].shape[0])
    print('Row count of Between 1982 and 1990: ', data_frames[3].shape[0])
    print('Row count of Before 1980: ', data_frames[4].shape[0])


def separate_data(data):
    # Separating the data according to the economic recessions in the USA.
    # Information about the recession dates retrieved from the site below:
    # https://en.wikipedia.org/wiki/List_of_recessions_in_the_United_States

    # after_2009=data[data['date'].dt.year>2009]
    after_2009 = filter_by_year(data, start=2009)
    # between_2001_07=data[(data['date'].dt.year<2007) & (data['date'].dt.year>2001)]
    between_2001_07 = filter_by_year(data, 2001, 2007)

    # between_91_2001=data[(data['date'].dt.year<2001) & (data['date'].dt.year>1991)]
    between_91_2001 = filter_by_year(data, 1991, 2001)

    # between_82_90=data[(data['date'].dt.year<1990) & (data['date'].dt.year>1982)]
    between_82_90 = filter_by_year(data, 1982, 1990)

    # before_80=data[data['date'].dt.year<1980]
    before_80 = filter_by_year(data, end=1980)

    return [after_2009, between_2001_07, between_91_2001, between_82_90, before_80]


def combine_datasets(data_frames):
    non_recessed_data = pd.concat(data_frames)
    non_recessed_data = non_recessed_data.sort_values(by='date')
    return non_recessed_data
