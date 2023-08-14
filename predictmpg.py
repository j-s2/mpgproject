import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from math import sqrt


# drop rows with '?'
test_data = pd.read_fwf(r'auto-mpg.data', skiprows=[32, 126, 330, 336, 354, 374], header=None)
test_data = test_data.drop(test_data.columns[8], axis=1) # drop last column 

# initialize linear regression 
linear = LinearRegression()

# scale columns that need scaling
# initialize scaler
minMax = MinMaxScaler()
# list all columns that do not need to be scaled
nonScaled = [0]
scaled_columns = test_data.columns[~test_data.columns.isin(nonScaled)]
# scale the data
test_data[scaled_columns] = minMax.fit_transform(test_data[scaled_columns])

# splits for cross validation
train, test = train_test_split(test_data, train_size = 0.7, test_size = 0.3, random_state = 10)

# set up feature selector
feats = SequentialFeatureSelector(linear, n_features_to_select=6, direction = "forward")
# fit the feature selector
feats.fit(test_data[scaled_columns], test_data[0])
# takes the list of the best features/columns that the feature selector found
predictors = list(scaled_columns[feats.get_support()])

# train model
linear.fit(train[predictors], train[0])

# test model
predictions = linear.predict(test[predictors])

# calculate RMSE
RMS = sqrt(mean_squared_error(test[0], predictions))
print(f"RMS: {RMS:>7f}")