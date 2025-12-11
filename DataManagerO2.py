import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt


dataset_locs = {
    'ETH/USD': "datasets\\ETHUSD_1.csv",
    'ETH/XBT': "datasets\\ETHXBT_1.csv",
    'XBT/USD': "datasets\\XBTUSD_1.csv",
    'XRP/USD': "datasets\\XRPUSD_1.csv",
    'XRP/ETH': "datasets\\XRPETH_1.csv",
    'XRP/XBT': "datasets\\XRPXBT_1.csv"
}


def print_earliest_data_avail():
    """Prints the earliest timestamp available for each ticker's dataset."""
    print('CHECKING EARLIEST AVAILABLE DATA:')
    for x in dataset_locs:
        df = pd.read_csv(dataset_locs[x]).set_index('time')
        d = df.head(1).index.values.astype(int)[0]
        print(x + ': ' + datetime.utcfromtimestamp(d).strftime("%m/%d/%Y, %H:%M:%S"))


def print_latest_data_avail():
    """Prints the latest timestamp available for each ticker's dataset."""
    print('CHECKING LATEST AVAILABLE DATA:')
    for x in dataset_locs:
        df = pd.read_csv(dataset_locs[x]).set_index('time')
        d = df.tail(1).index.values.astype(int)[0]
        print(x + ': ' + datetime.utcfromtimestamp(d).strftime("%m/%d/%Y, %H:%M:%S"))


def last_close_avail():
    """Returns the most recent closing price for ETH/USD."""
    df = pd.read_csv("datasets\\ETHUSD_1.csv").set_index('time')
    c = df['close'].values[-1]
    print(c)
    return c


def last_close_avail_sim(simTime, ticker):
    """Returns the closing price for a given ticker at a simulated time."""
    global c_ETHUSD, c_ETHXBT, c_XBTUSD, c_XRPUSD, c_XRPETH, c_XRPXBT

    # Check cached data for the ticker
    if ticker == 'ETH/USD' and c_ETHUSD is not None:
        data = c_ETHUSD
    elif ticker == 'ETH/XBT' and c_ETHXBT is not None:
        data = c_ETHXBT
    elif ticker == 'XBT/USD' and c_XBTUSD is not None:
        data = c_XBTUSD
    elif ticker == 'XRP/USD' and c_XRPUSD is not None:
        data = c_XRPUSD
    elif ticker == 'XRP/ETH' and c_XRPETH is not None:
        data = c_XRPETH
    elif ticker == 'XRP/XBT' and c_XRPXBT is not None:
        data = c_XRPXBT
    else:
        data = compiled_data(ticker)

    # Slice data around simTime and get last close
    try:
        df = data.loc[simTime-301:simTime].copy()
        if df.empty or len(df) == 0:
            print(f"[WARNING] No data for {ticker} in last 5 min @ {simTime}")
            return None  # ← CRITICAL: Return None if no data
        c = df['close'].values[-1]
        return c
    except Exception as e:
        print(f"Error in last_close_avail_sim for {ticker}: {e}")
        return None


def compileTargetTrend(unix_start, unix_end, c, target=True, close=False):
    """Compiles data for a ticker between unix_start and unix_end, optionally with trend labels.

    Args:
        unix_start (int): Start Unix timestamp.
        unix_end (int): End Unix timestamp.
        c (str): Ticker symbol (e.g., 'ETH/USD').
        target (bool): If True, compute trend labels based on future 4-hour mean with ATR-scaled threshold.
        close (bool): If True, include close price in output; otherwise, exclude it.

    Returns:
        pd.DataFrame: Processed DataFrame with indicators and optional trend/close.
    """
    df = compiled_data(c)
    df2 = df.loc[unix_start:unix_end].copy()
    df2 = df2.sort_values(by="time")

    if target:
        # Calculate trend based on future 360 minutes (6 hours)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=360)
        indexer2 = pd.api.indexers.FixedForwardWindowIndexer(window_size=720)
        # Compute next mean on sliced df2 to avoid index mismatch
        df2['nxtmean'] = df2['close'].rolling(window=indexer).mean()
        # Compute next max on sliced df2 to avoid index mismatch
        df2['nxtmax'] = df2['close'].rolling(window=indexer2).max()
        # Compute next max on sliced df2 to avoid index mismatch
        df2['nxtmin'] = df2['close'].rolling(window=indexer2).min()

        # Compute volatility-adjusted threshold using ATR
        # Use atr1 (14-minute ATR) and scale by its 1-day (1440-minute) rolling mean
        df2['avg_atr1'] = df2['atr1'].rolling(window=1440, min_periods=1).mean()
        # Avoid division by zero; use 1 if avg_atr1 is 0 or NaN
        df2['atr_scale'] = df2['atr1'] / df2['avg_atr1'].replace(0, 1)
        # Base threshold is 0.4% (0.004); scale it by ATR ratio, cap at reasonable bounds
        df2['threshold'] = 0.004 * df2['atr_scale'].clip(lower=0.5, upper=1.5)

        df2['trenddown'] = df2.apply(lambda x: (x.nxtmean <= (x.close - (x.close * x.threshold))), axis=1)

        # Base threshold off simple 0.3%
        # df2['threshold'] = 0.003

        # Calculate trend: up (1) if nxtmean exceeds close + threshold, down (-1) if below close - threshold
        df2['trendup'] = df2.apply(lambda x: (x.nxtmean >= (x.close + (x.close * x.threshold))), axis=1)
        df2['trenddown'] = df2.apply(lambda x: (x.nxtmean <= (x.close - (x.close * x.threshold))), axis=1)
        df2['trend'] = df2.apply(lambda x: (int(x.trendup) + (int(x.trenddown) * -1)), axis=1)

        # Remove temporary columns
        df2 = df2.drop(['nxtmean', 'nxtmax', 'nxtmin', 'avg_atr1', 'atr_scale', 'threshold', 'trendup', 'trenddown'], axis=1)

    # Select columns based on close parameter
    if close:
        df2 = df2.drop(['low', 'high', 'open', 'count'], axis=1)
    else:
        df2 = df2.drop(['low', 'high', 'open', 'close', 'count'], axis=1)

    # Append ticker to column names
    new_columns = list(map(lambda d: d + c, df2.columns))
    df2.columns = pd.Index(new_columns)

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    # print('\n\n\n\n')
    # print(str(df2.head(1)))

    return df2


c_ETHUSD = None
c_ETHXBT = None
c_XBTUSD = None
c_XRPUSD = None
c_XRPETH = None
c_XRPXBT = None


def compiled_data(ticker):
    """Loads and computes technical indicators for a given ticker, caching the result.

    Args:
        ticker (str): Ticker symbol (e.g., 'ETH/USD').

    Returns:
        pd.DataFrame: DataFrame with price data and computed indicators.
    """
    global c_ETHUSD, c_ETHXBT, c_XBTUSD, c_XRPUSD, c_XRPETH, c_XRPXBT

    # Return cached data if available
    if ticker == 'ETH/USD' and c_ETHUSD is not None:
        return c_ETHUSD
    elif ticker == 'ETH/XBT' and c_ETHXBT is not None:
        return c_ETHXBT
    elif ticker == 'XBT/USD' and c_XBTUSD is not None:
        return c_XBTUSD
    elif ticker == 'XRP/USD' and c_XRPUSD is not None:
        return c_XRPUSD
    elif ticker == 'XRP/ETH' and c_XRPETH is not None:
        return c_XRPETH
    elif ticker == 'XRP/XBT' and c_XRPXBT is not None:
        return c_XRPXBT

    # Load and sort data
    df = pd.read_csv(dataset_locs[ticker]).set_index('time')
    df = df.sort_values(by="time")

    # Compute Exponential Moving Averages (EMAs)
    df['dayema1'] = df['close'].ewm(span=60*24*12).mean()  # ~12 days
    df['dayema2'] = df['close'].ewm(span=60*24*26).mean()  # ~26 days
    df['dayema3'] = df['close'].ewm(span=60*24*9).mean()   # ~9 days
    df['hrema1'] = df['close'].ewm(span=60*12).mean()      # ~12 hours
    df['hrema2'] = df['close'].ewm(span=60*26).mean()      # ~26 hours
    df['hrema3'] = df['close'].ewm(span=60*9).mean()       # ~9 hours
    df['hrema4'] = df['close'].ewm(span=60*6).mean()       # ~6 hours
    df['hrema5'] = df['close'].ewm(span=60*3).mean()       # ~3 hours
    df['hrema6'] = df['close'].ewm(span=60).mean()       # ~1 hours

    # Compute Relative Strength Index (RSI)
    df['rsi1'] = rsi(df)                     # ~14 minutes
    df['rsi2'] = rsi(df, periods=14*60)      # ~14 hours
    df['rsi3'] = rsi(df, periods=14*1440)    # ~14 days

    # Compute Money Flow Index (MFI)
    df['mfi1'] = mfi(df['high'], df['low'], df['close'], df['volume'])         # ~14 minutes
    df['mfi2'] = mfi(df['high'], df['low'], df['close'], df['volume'], n=60*14)  # ~14 hours
    df['mfi3'] = mfi(df['high'], df['low'], df['close'], df['volume'], n=60*24*14)  # ~14 days

    # Compute price change deltas
    df['delta_p_1h'] = (df["close"] - df["open"].shift(60*1)) / df["open"].shift(60*1)      # 1 hour
    df['delta_p_8h'] = (df["close"] - df["open"].shift(60*8)) / df["open"].shift(60*8)      # 8 hours
    df['delta_p_day'] = (df["close"] - df["open"].shift(60*24)) / df["open"].shift(60*24)   # 1 day
    df['delta_p_week'] = (df["close"] - df["open"].shift(60*24*7)) / df["open"].shift(60*24*7)  # 1 week
    df['delta_p_month'] = (df["close"] - df["open"].shift(60*24*30)) / df["open"].shift(60*24*30)  # 1 month

    # Compute new indicators
    # MACD (Moving Average Convergence Divergence)
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = short_ema - long_ema
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']  # Histogram for divergence

    # Bollinger Bands (20-minute, 20-hour, 20-day)
    for period, prefix in [(20, 'bb1'), (20*60, 'bb2'), (20*1440, 'bb3')]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'{prefix}_upper'] = sma + (std * 2)
        df[f'{prefix}_lower'] = sma - (std * 2)
        df[f'{prefix}_mid'] = sma
        df[f'{prefix}_width'] = (df[f'{prefix}_upper'] - df[f'{prefix}_lower']) / df[f'{prefix}_mid']

    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift()).abs()
    df['tr3'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['true_range'] = df['true_range'].fillna(df['tr1'])  # Fill first NaN with high-low
    df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)  # Drop temporary columns
    df['atr1'] = df['true_range'].rolling(14, min_periods=1).mean()             # ~14 minutes
    df['atr2'] = df['true_range'].rolling(1460, min_periods=1).mean()          # ~14 hours
    df['atr3'] = df['true_range'].rolling(141440, min_periods=1).mean()        # ~14 days

    # # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    # print('true_range')
    # print(true_range)
    # print(str(len(true_range)))
    # print(pd.Series(true_range).rolling(14).mean() )


    # Stochastic Oscillator
    for period, prefix in [(14, 'stoch1'), (14*60, 'stoch2'), (14*1440, 'stoch3')]:
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        df[f'{prefix}_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'{prefix}_d'] = df[f'{prefix}_k'].rolling(window=3).mean()

    # Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))  # Minute-to-minute log returns

    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

    # Volume Deltas
    df['delta_v_1h'] = df['volume'] / df['volume'].shift(60)      # 1-hour volume change
    df['delta_v_8h'] = df['volume'] / df['volume'].shift(60*8)    # 8-hour volume change
    df['delta_v_day'] = df['volume'] / df['volume'].shift(60*24)  # 1-day volume change

    # Cache the DataFrame
    if ticker == 'ETH/USD':
        c_ETHUSD = df
    elif ticker == 'ETH/XBT':
        c_ETHXBT = df
    elif ticker == 'XBT/USD':
        c_XBTUSD = df
    elif ticker == 'XRP/USD':
        c_XRPUSD = df
    elif ticker == 'XRP/ETH':
        c_XRPETH = df
    elif ticker == 'XRP/XBT':
        c_XRPXBT = df

    return df


def rsi(df, periods=14, ema=True):
    """Calculates Relative Strength Index (RSI) for the 'close' column.

    Args:
        df (pd.DataFrame): DataFrame with 'close' column.
        periods (int): Number of periods for RSI calculation.
        ema (bool): If True, use exponential moving average; else, simple moving average.

    Returns:
        pd.Series: RSI values.
    """
    close_delta = df['close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema:
        ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    else:
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


# Calculate money flow index
# Contributed by Github member and chatgpt (thank you robot overlords)
def mfi(high, low, close, volume, n=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign

    # Calculate gain and loss using vectorized operations
    positive_mf = np.where(signed_mf > 0, signed_mf, 0)
    negative_mf = np.where(signed_mf < 0, -signed_mf, 0)

    mf_avg_gain = pd.Series(positive_mf).rolling(n, min_periods=1).sum()
    mf_avg_loss = pd.Series(negative_mf).rolling(n, min_periods=1).sum()

    return (100 - 100 / (1 + mf_avg_gain / mf_avg_loss)).to_numpy()
