import pandas as pd
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier




def create_nn_model(df):

    # Split into features and target

    X = df.drop(['trendup'], axis = 1)
    y = df['trendup'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

    #Scale data, otherwise model will fail.
    #Standardize features by removing the mean and scaling to unit variance
    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    #define the model
    #Experiment with deeper and wider networks
    model = Sequential()
    model.add(Dense(128, input_dim=14, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    #Output layer
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    # model.summary()

    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs =200, verbose=False)

    from matplotlib import pyplot as plt
    #plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    # acc = history.history['mean_absolute_error']
    # val_acc = history.history['val_mean_absolute_error']
    # plt.plot(epochs, acc, 'y', label='Training MAE')
    # plt.plot(epochs, val_acc, 'r', label='Validation MAE')
    # plt.title('Training and validation MAE')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # # plt.show()

    ############################################
    #Predict on test data
    predictions = model.predict(X_test_scaled[:5])
    print("Predicted values are: ", predictions)
    print("Real values are: ", y_test[:5])
    ##############################################

    #Comparison with other models..
    #Neural network - from the current code
    mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
    print('Mean squared error from neural net: ', mse_neural)
    print('Mean absolute error from neural net: ', mae_neural)

    return model


def create_rf_model(df):

    # Split into features and target

    X = df.drop(['trend'], axis = 1)

    y = df['trend'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Scale data, otherwise model will fail.
    #Standardize features by removing the mean and scaling to unit variance
    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ##############################################
    #Random forest.
    #Increase number of tress and see the effect
    model = RandomForestRegressor(n_estimators = 40, verbose=0)
    model.fit(X_train_scaled, y_train)

    y_pred_RF = model.predict(X_test_scaled)

    predictions = model.predict(X_test_scaled[:5])
    print("Predicted values are: ", predictions)
    print("Real values are: ", y_test[:5])

    mse_RF = mean_squared_error(y_test, y_pred_RF)
    mae_RF = mean_absolute_error(y_test, y_pred_RF)
    print('Updated Model...')
    print('Mean squared error using Random Forest: ', mse_RF)
    print('Mean absolute error Using Random Forest: ', mae_RF)

    #Feature ranking...
    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    # print(feature_imp)

    return model

# tickers = {'ETH/USD':"", 'ETH/XBT':"", 'XBT/USD':"", 'XRP/USD':"", 'XRP/ETH':"", 'XRP/XBT':""}

def create_rfc_model(df, ticker, tickers):

    # remove unneeded target variables
    for key in tickers.keys():
        # print(ticker)
        # print(key)
        if key != ticker:
            # print('hey')
            df = df.drop(['trend' + key], axis = 1)

    X = df.drop(['trend' + ticker], axis = 1)

    y = df['trend' + ticker].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)

    #Scale data, otherwise model will fail.
    #Standardize features by removing the mean and scaling to unit variance
    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ##############################################
    #Random forest.
    #Increase number of tress and see the effect
    model = RandomForestClassifier(n_estimators = 80, verbose=0, warm_start=True)
    model.fit(X_train_scaled, y_train)

    y_pred_RF = model.predict(X_test_scaled)

    predictions = model.predict(X_test_scaled[:5])

    accuracy = accuracy_score(y_test, y_pred_RF)
    # print("Accuracy:", accuracy)

    # print("Predicted values are: ", predictions)
    # print("Real values are: ", y_test[:5])
    #
    # mse_RF = mean_squared_error(y_test, y_pred_RF)
    # mae_RF = mean_absolute_error(y_test, y_pred_RF)
    # print('Updated Model...')
    # print('Mean squared error using Random Forest: ', mse_RF)
    # print('Mean absolute error Using Random Forest: ', mae_RF)

    #Feature ranking...
    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    # print(feature_imp)

    return model
