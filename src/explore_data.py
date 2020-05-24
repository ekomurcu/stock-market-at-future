def explore_data(data, verbose=1):
    if verbose:
        # Giving basic information regarding the dataset such as shape, data types and descriptive statistics that
        # summarize columns
        print("Data Dimensionality: ", data.shape)
        print("There are " + str(data.shape[0]) + " rows and " + str(
            data.shape[1]) + " attributes in historical stock prices data")

        # print attribute names
        print("Attribute Names: ", data.columns)

        print("The number of stocks are: " + str((data['ticker'].nunique())))

        # print first 5 rows in your dataset
        print("Head of Data: ")
        print(data.head())

        # print data types of the columns
        print("Data types: ")
        print(data.dtypes)

        print("NaN counts in each column: ")
        print(data.isnull().sum())  # NaN counts in each column
        print("There is no missing value at any attribute or row so we are fine to proceed without preprocessing.")

        # Descriptive statistics of taxitrips data
        print("Descriptive statistics of taxitrips data:")
        print(data.describe())
