
def explore_data(data):
    # Giving basic information regarding the dataset such as shape, data types and descriptive statistics that summarize columns
    # print shape
    print('Data Dimensionality: ', data.shape)
    print("There are " + str(data.shape[0]) + " rows and " + str(data.shape[1]) + " attributes in historical stock prices data")

    # print attribute names
    print('Attribute Names: ', data.columns)

    print("The number of stocks are: " + str((data['ticker'].nunique())))

    # print first 5 rows in your dataset
    print('Head of Data: ')
    data.head()

    # print data types of the columns
    print('Data types: ', data.dtypes)

    data.isnull().sum()  # NaN counts in each column

    # Descriptive statistics of taxitrips data
    data.describe()
