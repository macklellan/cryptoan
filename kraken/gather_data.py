#
# Simple data gathering program. Uses Kraken Exchange API.
# Gathers data for an amount of time and exports to a csv file
# I love Nolan
#
import numpy as np
import pandas as pd
import time
import json
import krakenex
from datetime import datetime

k = krakenex.API()

#k.load_key('kraken.key')

# KRAKEN TICKER RESPONSE DESCRIPTION
# a = ask array(<price>, <whole lot volume>, <lot volume>)
# b = bid array(<price>, <whole lot volume>, <lot volume>)
# c = last trade closed array(<price>, <lot volume>)
# v = volume array(<today>, <last 24 hours>)
# p = volume weighted average price array(<today>, <last 24 hours>)
# t = number of trades array(<today>, <last 24 hours>)
# l = low array(<today>, <last 24 hours>)
# h = high array(<today>, <last 24 hours>)
# o = today’s opening price


# dataframe for storing kraken market ticker responses
df = pd.DataFrame(columns=["a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "v1", "v2", "p1", "p2", "t1", "t2", "l1", "l2", "h1", "h2", "o", "time"])

try:

    # Will run for approximately 12 hours if each loop takes 1 minutes.
    for i in range(0, 720):

        # requesting ticker data for DOGE to USD market
        response = k.query_public('Ticker', {'pair': 'DOGEUSD'})

        data = response['result']['XDGUSD']

        row = [ data['a'][0], data['a'][1], data['a'][2], data['b'][0], data['b'][1], data['b'][2], data['c'][0], data['c'][1], data['v'][0], data['v'][1], data['p'][0], data['p'][1], data['t'][0], data['t'][1], data['l'][0], data['l'][1], data['h'][0], data['h'][1], data['o'], datetime.utcnow() ]

        # appends row to dataframe
        df.loc[len(df)] = row

        # waiting for 1 minute
        time.sleep(60)


    # exporting data to csv file
    df.to_csv('datatest.csv')
    print('Finished gathering data!')


except Exception as e:
    print(str(e))
    df.to_csv('datatest.csv')
