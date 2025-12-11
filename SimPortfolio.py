# Ryan McClellan

class SimPortfolio:

    def __init__(self, cryptos, startingUSD, **kwargs):

        self.balances = {}
        unique_symbols = set()
        for pair in cryptos:
            sym1, sym2 = pair.split('/')
            unique_symbols.add(sym1)
            unique_symbols.add(sym2)

        for sym in unique_symbols:
            self.balances[sym] = 0.0

        if 'USD' in self.balances:
            self.balances['USD'] = startingUSD
        else:
            raise ValueError("USD not found in ticker symbols. Cannot set starting balance.")

        self.costs = {}
        self.trades = {}

        for key, value in kwargs.items():
            if key == 'debug' and value == True:
                self.debug = True

        self.printbalances()

    def printbalances(self, close_all_tickers=None):
        print('BALANCES: ')
        print(self.balances)

        if close_all_tickers is not None:
            tot = 0.0
            for sym, bal in self.balances.items():
                if bal <= 0:
                    continue
                if sym == 'USD':
                    tot += bal
                elif sym + '/USD' in close_all_tickers:
                    tot += bal * close_all_tickers[sym + '/USD']
                elif sym + '/XBT' in close_all_tickers and 'XBT/USD' in close_all_tickers:
                    tot += bal * close_all_tickers[sym + '/XBT'] * close_all_tickers['XBT/USD']
                else:
                    print(f"Warning: No direct or indirect USD conversion found for {sym}")
            print('TOTAL VALUE: ' + str(tot))


    def calcTotal(self, close_all_tickers=None):

        if close_all_tickers is not None:
            tot = 0.0
            for sym, bal in self.balances.items():
                if bal <= 0:
                    continue
                if sym == 'USD':
                    tot += bal
                elif sym + '/USD' in close_all_tickers:
                    tot += bal * close_all_tickers[sym + '/USD']
                elif sym + '/XBT' in close_all_tickers and 'XBT/USD' in close_all_tickers:
                    tot += bal * close_all_tickers[sym + '/XBT'] * close_all_tickers['XBT/USD']
                else:
                    print(f"Warning: No direct or indirect USD conversion found for {sym}")
            return tot


    def updateRecord(self, row, total):
        s = '\n' + str(row[0]) + ', ' + str(row[1]) + ', ' + str(row[2]) + ', ' + str(row[3]) + ', ' + str(row[4]) + ', ' + str(row[5]) + ', $' + str(total)
        with open('record.csv','a') as fd:
            fd.write(s)

    def quickBuy(self, name, time, pair, amount, price, tickers):
        symbols = pair.split("/")
        self.balances[symbols[0]] += amount
        self.balances[symbols[1]] -= amount * price
        self.costs[time] = (time, symbols[0], symbols[1], price, amount)
        self.updateRecord((time, symbols[0], symbols[1], price, amount, name), self.calcTotal(tickers))

    def quickSell(self, name, time, pair, amount, price, tickers):
        symbols = pair.split("/")
        self.balances[symbols[0]] -= amount
        self.balances[symbols[1]] += amount * price
        self.updateRecord((time, symbols[1], symbols[0], price, amount, name), self.calcTotal(tickers))

    def lastBuy(self, pair):
        symbols = pair.split("/")
        last = (0, 0, 0, 0, 0)

        for time, order in self.costs.items():
            # check symbols match
            if order[1] == symbols[0] and order[2] == symbols[1]:
                # later time
                if time > last[0]:
                    last = order

        return last

    def lastSell(self, pair):
        symbols = pair.split("/")
        last = (0, 0, 0, 0, 0)

        for time, order in self.costs.items():
            # check symbols match
            if order[1] == symbols[1] and order[2] == symbols[0]:
                # later time
                if time > last[0]:
                    last = order

        return last

    def lastOrder(self, pair):

        lastB = self.lastBuy(pair)
        lastS = self.lastSell(pair)

        if lastB[0] > lastS[0]:
            return lastB
        else:
            return lastS
