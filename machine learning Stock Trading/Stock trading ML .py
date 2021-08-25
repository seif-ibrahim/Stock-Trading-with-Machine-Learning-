from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# evaluation metric import
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error

# read data
data = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv')
# df2 = pd.read_csv('AMZN_2006-01-01_to_2018-01-01.csv')
# df3 = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv')

# data = pd.concat([df1, df2, df3], ignore_index=True) #concat three of dataset

# data modeling
data['Date'] = pd.to_datetime(data['Date'])  # transform to datetime datatype
data['Volume'] = data['Volume'].astype(float)  # make Volume column float

data['Year'], data["Month"], data['Days'] = data.Date.dt.year, data.Date.dt.month, data.Date.dt.day
# but year and month and day in a specific column

# print(data.shape)
# print(data.head())
# print(data.describe())
# print(data.info())

##data modeling
my_feature = ['Year', 'Month', 'Days', "Open", 'High', 'Low', "Volume"]  # feature
my_target = ['Close']  # target

my_features = data[my_feature].values
my_target = data[my_target].values

x_train, x_test, y_train, y_test = train_test_split(my_features, my_target, test_size=0.25,
                                                    random_state=0)  # split to 25 % test values and 75 % train

sc = MinMaxScaler(feature_range=(0, 1))  # normalize the data between 0 and 1 as default
x_train_scaled = sc.fit_transform(x_train)  # train
x_train_df = pd.DataFrame(x_train_scaled)  # x train

x_test_scaled = sc.fit_transform(x_test)  # test
x_test_df = pd.DataFrame(x_test_scaled)  # x test

my_model = DecisionTreeRegressor(max_depth=3)  # max depth of the DecisionTree is 3

my_model.fit(x_train_df, y_train)  # train the model

y_pred = my_model.predict(x_test_df)    # predict the test data
y_test = y_test.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)
#y_pred = sc.inverse_transform(y_pred)      # to make it as normal numbers without normalize
print("DecisionTreeRegressor with depth 3 score",
      my_model.score(x_test, y_test))  # its a built in function that give the score of the model

# visualization
plt.figure(figsize=(16, 8))
plt.title('model')
plt.xlabel('Date', fontsize=50)
plt.ylabel('Close Price USD ($)', fontsize=50)
plt.plot(y_test[:], color='red', label='Actual Close Price')
plt.plot(y_pred[:], color='green', label='Predicted Price')
plt.legend(loc='lower right')
plt.show()

# regression evaluation metrics

print("DecisionTreeRegressor with depth 3 and R^2 is =",
      r2_score(y_test,
               y_pred))  # its R2 evaluation different between the predict and the true of the predict can be Negative

print("DecisionTreeRegressor with depth 3 and Mean_absolute error is =",
      mean_absolute_error(y_test,
                          y_pred))
# expected value of the absolute error loss average of every sample error(actuale value - predicted value)

print("DecisionTreeRegressor with depth 3 and Mean 2 error is =",
      mean_squared_error(y_test, y_pred))  # the expected value of the squared error or loss.

print("DecisionTreeRegressor with depth 3 and Variance_score is = ",
      explained_variance_score(y_test,
                               y_pred))
# its R2 evaluation different between the predict and the true of the predict but with variance equation

#   multioutput{‘raw_values’, ‘uniform_average’, ‘variance_weighted’}
# ‘raw_values’ :
# Returns a full set of scores in case of multioutput input.
#
# ‘uniform_average’ :
# Scores(errors at mean_squared_error) of all outputs are AVERAGE with uniform weight.
#
# ‘variance_weighted’ : not in mean_squared_error
# Scores of all outputs are averaged, weighted by the variances of each individual output.
print("DecisionTreeRegressor with depth 3 and Max error is =",
      max_error(y_test, y_pred))  # metric that calculates the maximum residual error.
