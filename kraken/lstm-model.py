#
# Long Short Term Memory (LSTM) model
#
# NOT WORKING YET
#
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#creating dataframe fro9m csv data
data = pd.read_csv('data/doge-USD.csv')

#remove index attribute
data.drop('n', axis=1, inplace=True)

dependent = data.loc[:,'a1'].to_numpy()

print(data.head(5))

#setting index
data.index = data.time

#creating train and test sets
dataset = data.values
train = dependent[:350]
valid = dependent[351:]

# # specify columns to plot
# groups = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# i = 1
# # plot each column
# plt.figure()
# for group in groups:
# 	plt.subplot(len(groups), 1, i)
# 	plt.plot(dataset[:, group])
# 	plt.title(data.columns[group], y=0.5, loc='right')
# 	i += 1
# plt.show()



# #converting dataset into x_train and y_train
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(dataset)
#
# x_train, y_train = [], []
# for i in range(60,len(train)):
#     x_train.append(scaled_data[i-60:i,0])
#     y_train.append(scaled_data[i,0])
# x_train, y_train = np.array(x_train), np.array(y_train)
#
# x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))
#
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
#
# #predicting 246 values, using past 60 from the train data
# inputs = new_data[len(new_data) - len(valid) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs  = scaler.transform(inputs)
#
# X_test = []
# for i in range(60,inputs.shape[0]):
#     X_test.append(inputs[i-60:i,0])
# X_test = np.array(X_test)
#
# X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# closing_price = model.predict(X_test)
# closing_price = scaler.inverse_transform(closing_price)
