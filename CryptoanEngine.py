#
# Ryan McClellan 11/2023

from datetime import datetime, timedelta
import time
from CryptoanBotSt import CryptoanBotSt
from CryptoanBotLt import CryptoanBotLt
import pandas as pd
import keyboard


class CryptoanEngine:

    def __init__(self, **kwargs):

        # set start date of simulation
        self.startDatetime = datetime.strptime("2024-11-01 12:00:00", "%Y-%m-%d %H:%M:%S")
        self.endDatetime = datetime.strptime("2024-12-01 12:00:00", "%Y-%m-%d %H:%M:%S")
        self.startUnixtime = time.mktime(self.startDatetime.timetuple())
        self.endUnixtime = time.mktime(self.endDatetime.timetuple())
        self.runningTime = self.startUnixtime
        self.simInterval = 5 # minutes


        self.stTickers = {'ETH/USD':"", 'ETH/XBT':"", 'XBT/USD':"", 'LTC/XBT':"", 'XMR/XBT':""}
        self.ltTickers = {'ADA/USD':"", 'ADA/XBT':"", 'ETH/USD':"", 'ETH/XBT':"", 'SOL/ETH':"", 'SOL/USD':"", 'SOL/XBT':""}

        self.bots = {}

        for key, value in kwargs.items():
            if key == 'debug' and value == True:
                self.debug = True



    def startTradeSimulationSimTime(self):


        print('SHORT TERM TRADING TICKERS: ' + str(self.stTickers))
        print('LONG TERM TRADING TICKERS: ' + str(self.ltTickers))
        print('SIMULATED TIME TRADE SIMULATION. STARTING.. @ .' + self.startDatetime.strftime("%m/%d/%Y, %H:%M:%S"))
        print('Creating Bots....')

        self.bots['ST1'] = CryptoanBotSt('ST1', self.stTickers, time=self.runningTime, xPredTime=1)
        self.bots['ST2'] = CryptoanBotSt('ST2', self.stTickers, time=self.runningTime, xPredTime=.8)
        self.bots['ST3'] = CryptoanBotSt('ST3', self.stTickers, time=self.runningTime, xPredTime=1.5)
        self.bots['LT1'] = CryptoanBotLt('LT1', self.ltTickers, time=self.runningTime, xPredTime=1)
        self.bots['LT2'] = CryptoanBotLt('LT2', self.ltTickers, time=self.runningTime, xPredTime=.8)
        self.bots['LT3'] = CryptoanBotLt('LT3', self.ltTickers, time=self.runningTime, xPredTime=1.5)


        while self.runningTime < self.endUnixtime:

            self.runningTime += self.simInterval

            try:

                for key in self.bots.keys():

                    self.bots[key].updateTime(self.runningTime)
                    self.bots[key].doWork()

                if keyboard.is_pressed("q"):
                    s = datetime.utcfromtimestamp(self.runningTime).strftime("%m/%d/%Y, %H:%M:%S")
                    print(s)
                    time.sleep(5)
                    break

            except ValueError as e:
                print(e)
                # raise(e)
                self.runningTime += self.simInterval*60
            except Exception as e:
                print(e)
                raise(e)
                self.runningTime += self.simInterval*60


    def test(self):

        print('LONG TERM TRADING TICKERS: ' + str(self.ltTickers))

        self.bots['LT1'] = CryptoanBotLt('LT1', self.ltTickers, time=self.runningTime, xPredTime=1)

        self.bots['LT1'].test()
