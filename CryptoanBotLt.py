#
# Ryan McClellan 11/2023

from datetime import datetime, timedelta
import time
# from TradeManager import TradeManager
# from SimTradeManager import SimTradeManager
from ModelManager import create_rf_model, create_rfc_model, create_nn_model
from SimPortfolio import SimPortfolio
from sklearn.preprocessing import StandardScaler
import DataManager15
import pandas as pd
import keyboard
import os


class CryptoanBotLt:

    def __init__(self, name, tickers, **kwargs):

        print("(Cryptoan) Initialized long term trading bot...  Name: " + name)

        self.name = name
        self.runningTime = None
        self.lastRunTime = 0
        self.modelCreateTime = 0
        self.tickers = tickers
        self.predMulti = 1
        self.tickersList = self.tickers.keys()
        self.data_locs = DataManager15.set_dataset_locations(self.tickersList)
        self.models = {}
        self.scaler = None
        self.portfolio = SimPortfolio(cryptos=self.tickersList, startingUSD=1000)

        for key, value in kwargs.items():
            if key == 'debug' and value == True:
                self.debug = True
            if key == 'time':
                self.runningTime = value
            if key == 'xPredTime':
                self.predMulti = value


    def updateTime(self, time):
        self.runningTime = time

    def doWork(self):

        # limit predictions to once every 15 minutes
        if self.runningTime >= (self.lastRunTime + 15*60):
            self.lastRunTime = self.runningTime
            self.__predictionTrader2()
            self.__stopLossTrader()


    def test(self):

        self.lastRunTime = self.runningTime
        trainingStart = self.runningTime - (86400 * 365)  # 365 days
        trainingEnd = self.runningTime - 3601
        print('Start: ' + str(trainingStart))
        print('End: ' + str(trainingEnd))
        self.__exportCompoundData(trainingStart, trainingEnd)


    def __predictionTrader2(self):

        # get predictions
        outlook = self.__getOutlook()

        # last close for all tickers
        close = {}
        for pair in self.tickers.keys():
            close[pair] = DataManager15.last_close_avail_sim(self.runningTime, pair)

        # loop predictions
        for pair in outlook.keys():
            symbols = pair.split("/")

            # prediction price will rise
            if outlook[pair] == True:

                # check purchasing balance is not 0
                if self.portfolio.balances[symbols[1]] <= 0:
                    break
                self.portfolio.quickBuy(self.name, self.runningTime, pair, (self.portfolio.balances[symbols[1]]/close[pair]), close[pair], close)
                print('(' + self.name + ') PREDICTION BUY: ' +  str(pair) + ' @ ' + '$' + str(close[pair]) )
                self.portfolio.printbalances(close)

            # prediction price will sink
            elif outlook[pair] == False:

                # check selling balance not 0
                if self.portfolio.balances[symbols[0]] <= 0:
                    break
                self.portfolio.quickSell(self.name, self.runningTime, pair, self.portfolio.balances[symbols[0]], close[pair], close)
                print('(' + self.name + ') PREDICTION SELL: ' +  str(pair) + ' @ ' + '$' + str(close[pair]) )
                self.portfolio.printbalances(close)



    def __stopLossTrader(self):

        mvmnt = self.__recent()


        # last close for all tickers
        close = {}
        for pair in self.tickers.keys():
            close[pair] = DataManager15.last_close_avail_sim(self.runningTime, pair)

        stoploss = -0.05
        stopgain = .01

        for pair in mvmnt.keys():

            symbols = pair.split("/")

            if mvmnt[pair] > 0 and mvmnt[pair] > stopgain:
                self.portfolio.quickSell(self.name, self.runningTime, pair, self.portfolio.balances[symbols[0]], close[pair], close)
                print('(' + self.name + ') STOPGAIN SELL: ' +  str(pair) + ' @ ' + '$' + str(close[pair]) )
                self.portfolio.printbalances(close)
            # stoploss sale
            if mvmnt[pair] < 0 and mvmnt[pair] < stoploss:
                self.portfolio.quickSell(self.name, self.runningTime, pair, self.portfolio.balances[symbols[0]], close[pair], close)
                print('(' + self.name + ') STOPLOSS SELL @: ' +  str(pair) + ' @ ' + '$' + str(close[pair]) )
                self.portfolio.printbalances(close)



    def __getOutlook(self):


        try:

            # update expired (24hrs) or missing ml model
            if self.modelCreateTime < self.runningTime-86400:
                self.__updateMLModel()

            preds = self.__makePredictions()


            out = {}

            for key in self.tickers.keys():

                if preds[key] == 1:
                    out[key] = True
                elif preds[key] == -1:
                    out[key] = False

            return out


        except Exception as e:
            raise e


    def __getCompoundData(self, trainingStart, trainingEnd, target=True):

        data_d = {}

        data = pd.DataFrame()  # Still start empty, but we'll handle the first iteration differently

        first = True

        for key in self.tickers.keys():
            data_d[key] = DataManager15.compileTargetTrend(trainingStart, trainingEnd, key, target, False, self.predMulti)  # Fixed: Use 'key' instead of hardcoded 'ETH/USD'
            # data_d[key] = data_d[key].dropna()
            # print("TRAINING DATA LENGTH (" + key + ") ->  " + str(len(data_d[key].index)))

            # if target:
            #     print("TRAINING DATA LENGTH (" + key + ") ->  " + str(len(data_d[key].index)))  # Optional: Made dynamic, but commented out as in original

            if first:
                data = data_d[key]
                first = False
            else:
                data = pd.merge_asof(data, data_d[key], on='time')

        # if target == False:
        #     pd.set_option('display.max_columns', None)
        #     pd.set_option('display.max_colwidth', None)
        #     print(data.head(1))
        #     print(list(data))

        data = data.dropna()

        # print(data.head(1))
        # print(list(data))

        return data


    def __exportCompoundData(self, trainingStart, trainingEnd, target=True):

        data_d = {}

        data = pd.DataFrame()  # Still start empty, but we'll handle the first iteration differently

        first = True

        for key in self.tickers.keys():
            data_d[key] = DataManager15.compileTargetTrend(trainingStart, trainingEnd, key, target, False, self.predMulti)  # Fixed: Use 'key' instead of hardcoded 'ETH/USD'
            # data_d[key] = data_d[key].dropna()
            # print("TRAINING DATA LENGTH (" + key + ") ->  " + str(len(data_d[key].index)))

            # if target:
            #     print("TRAINING DATA LENGTH (" + key + ") ->  " + str(len(data_d[key].index)))  # Optional: Made dynamic, but commented out as in original

            if first:
                data = data_d[key]
                first = False
            else:
                data = pd.merge_asof(data, data_d[key], on='time')

        data = data.dropna()

        # Define output file path
        # output_file = os.path.join(root_dir, "output.csv")
        # Save to CSV without index, ensuring correct columns
        data.to_csv("output.csv", index=False, header=True)


    def __updateMLModel(self):
        print('(' + self.name + ') Start Updated ML Models')
        trainingStart = self.runningTime - (86400 * 100)  # 365 days
        trainingEnd = self.runningTime - 3601
        compoundData = self.__getCompoundData(trainingStart, trainingEnd)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_colwidth', None)
        # print(compoundData.head(1))
        if not compoundData.empty:

            # re-fit scaler
            self.scaler = StandardScaler().fit(compoundData.drop(columns=[col for col in compoundData if 'trend' in col]))

            for key in self.tickers:
                self.models[key] = create_rfc_model(compoundData, key, self.tickers)

            self.modelCreateTime = self.runningTime

            s = datetime.utcfromtimestamp(self.runningTime).strftime("%m/%d/%Y, %H:%M:%S")

            print('(' + self.name + ') Updated ML Models @ ' + str(s))


    def __makePredictions(self):


        # get latest data (15 minutes)
        T2 = self.__getCompoundData(self.runningTime-1801, self.runningTime, False)

        # print('T2')
        # print(T2)

        # get latest data row
        X2 = T2.tail(1)

        # normalize / scale input data
        X_test_scaled2 = self.scaler.transform(X2.drop(columns=[col for col in X2 if 'trend' in col], errors='ignore'))

        y_preds = {}

        # loop tickers and store predictions for each
        for key in self.tickers.keys():
            y_preds[key] = int(self.models[key].predict(X_test_scaled2)[0])

        return y_preds



    def __recent(self):

        tickers = self.tickers
        movement = {}
        for key in self.tickers.keys():
            movement[key] = 0

        close = {}

        for pair in tickers.keys():
            close[pair] = DataManager15.last_close_avail_sim(self.runningTime, pair)

        # available balances
        for key in self.portfolio.balances.keys():
            if self.portfolio.balances[key] > 0:

                # all possible trading pairs
                for key2 in self.portfolio.balances.keys():
                    if key != key2:
                        pair = key + '/' + key2
                        lastBuy = self.portfolio.lastBuy(pair)

                        # last order
                        if lastBuy[0] != 0:
                            movement[pair] = (close[pair] - lastBuy[3]) / lastBuy[3]


        # print("LAST CLOSE")
        # print(close)
        # print("MVMNT")
        # print(movement)

        return movement
