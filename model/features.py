import concurrent.futures
import time
from concurrent.futures import as_completed

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

import util.loggers as loggers

logger = loggers.setup_loggers()
tradex_logger = logger['tradex']


class Tradex_indicator:
    '''
    This class applies various trading indicators on given data. The data can either be provided
    during the initialization or fetched from an external source.

    Attributes:
        symbol (str): Symbol for which the data is to be processed
        data (DataFrame): Data to be processed. The dataframe should contain the columns 'open', 'high', 'low', 'close' and 'volume'
        trend (Trend): Trend object after processing the data
        screener (type): Description of attribute `screener`.
        real_time (type): Description of attribute `real_time`.
        scanner (type): Description of attribute `scanner`.
    '''

    def __init__(self, symbol, timeframe, t=None, get_data=False, data=None):
        self.tradex_logger = tradex_logger
        self.ccxt_exchange = ccxt.binance()  # Change this to your desired exchange
        self.timeframe = timeframe
        self.symbol = symbol
        self.data = data if not get_data else self.get_data()
        if t is not None:
            self.changeTime(t)
        self.trend = Trend
        self.screener = Screener
        self.real_time = Real_time
        self.scanner = Scanner

    @staticmethod
    def convert_df(df):
        # List of columns to be converted to float
        columns_to_convert = ['open', 'high', 'low', 'close', 'volume']

        # Apply the conversion to the specified columns
        df[columns_to_convert] = df[columns_to_convert].astype(np.int64)
        df[columns_to_convert] = df[columns_to_convert].round(6)

        return df

    def get_data(self) -> pd.DataFrame:
        try:
            self.tradex_logger.info('Getting the data')
            since = self.ccxt_exchange.parse8601(
                (pd.Timestamp('3 days ago')).isoformat())
            data_load = self.ccxt_exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, since=since)

            df = pd.DataFrame(data_load, columns=[
                              'date', 'open', 'high', 'low', 'close', 'volume'])
            df = pd.to_numeric(df, errors='coerce')
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df.set_index('date', inplace=True)
            df['symbol'] = self.symbol

        except ccxt.NetworkError as e:
            self.tradex_logger.error(f'Network error: {e}')
            return None

        except ccxt.ExchangeError as e:
            self.tradex_logger.error(f'Exchange error: {e}')
            return None

        except Exception as e:
            self.tradex_logger.error(f'An unexpected error occurred: {e}')
            return None

        return df

    def changeTime(self, t):
        try:
            self.data.index = pd.to_datetime(self.data.index, utc=True)
            self.tradex_logger.info(f'Changing the {self.timeframe} to {t}')
            ohlc = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            self.data = self.data.resample(
                t, label='left', kind='timestamp').apply(ohlc).dropna()

        except pd.errors.EmptyDataError:
            self.tradex_logger.error(
                'Empty data received while changing the timeframe.')
            return None

        except pd.errors.OutOfBoundsDatetime:
            self.tradex_logger.error(
                'Error encountered in the datetime bounds while changing the timeframe.')
            return None

        except pd.errors.OutOfBoundsTimedelta:
            self.tradex_logger.error(
                'Error encountered in the timedelta bounds while changing the timeframe.')
            return None

        except pd.errors.ResampleError:
            self.tradex_logger.error(
                'Error encountered while resampling the data.')
            return None

        except Exception as e:
            self.tradex_logger.error(
                f'An unexpected error occurred while changing the timeframe: {e}')
            return None

    def run(self):
        try:
            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                trend = executor.submit(Trend, self.data, self.tradex_logger)
                screener = executor.submit(
                    Screener, self.data, self.tradex_logger)
                real_time = executor.submit(
                    Real_time, self.data, self.tradex_logger)
                scanner = executor.submit(
                    Scanner, self.data, self.tradex_logger)

            results = {result.result().__str__(): result.result()
                       for result in as_completed([trend, screener, real_time, scanner])}

            self.trend = results.get('trend', None)
            self.screener = results.get('screener', None)
            self.real_time = results.get('real_time', None)
            self.scanner = results.get('scanner', None)

            if not any([self.trend, self.screener, self.real_time, self.scanner]):
                self.tradex_logger.info("Error - no indicator found!")

            end_time = time.time()
            self.tradex_logger.info(
                f"Elapsed time: {end_time - start_time:.2f} seconds")
            self.tradex_logger.info("_"*20)

        except concurrent.futures.TimeoutError:
            self.tradex_logger.error(
                "Timeout occurred while executing the analysis threads.")
            return None

        except Exception as e:
            self.tradex_logger.error(
                f"An unexpected error occurred during the analysis: {e}")
            return None


class Trend:
    '''
    TREND: visueel beeld van de market trend
    '''

    def __init__(self, data, tradex_logger):
        self.tradex_logger = tradex_logger
        self.data = data
        self.df_trend = pd.DataFrame()
        self.get_trend()

    # TRADE - X TREND

    def get_trend(self) -> pd.DataFrame:
        self.tradex_logger.info('init trade-x trend')

        # EMA channel
        ema55H, ema55L = self.hlChannel()

        # EMA trend lines
        ema_200, ema_100 = self.ema()

        # lsma and ema
        lsma, ema_10 = self.lsma_()

        # vwap and wma
        vwap, wma = self.vwap_()

        golden_signal = np.where(ta.cross(wma, vwap), 1, 0)
        golden_signal = np.where(ta.cross(vwap, wma), -1, golden_signal)

        # stoploss
        stop_loss = self.stoploss()

        # adding the data to the trend dataframe

        self.df_trend['ema55H'] = ema55H
        self.df_trend['ema55L'] = ema55L
        self.df_trend['ema_100'] = ema_100
        self.df_trend['ema_200'] = ema_200
        self.df_trend['lsma'] = lsma
        self.df_trend['ema_10'] = ema_10
        self.df_trend['vwap'] = vwap
        self.df_trend['wma'] = wma
        self.df_trend['golden_signal'] = golden_signal

        self.df_trend['stop_loss'] = stop_loss

        # adding the data to the general dataframe
        self.data['ema55H'] = self.df_trend.ema55H
        self.data['ema55H'] = self.df_trend.ema55L
        self.data['ema_100'] = self.df_trend.ema_100
        self.data['ema_200'] = self.df_trend.ema_200
        self.data['lsma'] = self.df_trend.lsma
        self.data['ema_10'] = self.df_trend.ema_10
        self.data['golden_signal'] = golden_signal
        self.data['vwap'] = vwap
        self.data['wma'] = wma

        return self.df_trend

    def hlChannel(self):
        self.tradex_logger.info('- setting up High|Low channel')

        # create the 55 ema channel
        ema55H = ta.ema(self.data['high'], 55)
        ema55L = ta.ema(self.data['low'], 55)

        return ema55H, ema55L

    def ema(self):
        self.tradex_logger.info('- setting up EMA')

        # create the ema trend
        ema_200 = ta.ema(self.data['close'], 200)
        ema_100 = ta.ema(self.data['close'], 100)

        return ema_200, ema_100

    def lsma_(self):
        self.tradex_logger.info('- setting up LSMA')

        lsma = ta.linreg(self.data['low'], 20, 8)
        ema_10 = ta.ema(self.data['close'], 10)

        return lsma, ema_10

    def vwap_(self):
        self.tradex_logger.info('- setting up VWAP')

        vwma = ta.vwma(self.data['close'], self.data['volume'], 14)

        wma = ta.wma(vwma, 21)

        vwap = self.calculate_vwap(wma, self.data['volume'])

        # Calculate the VWAP using the "price" and "volume" columns
        # vwap = (df["price"] * df["volume"]).sum() / df["volume"].sum()

        return vwap, wma

    def get_tv_vwap(self, source, volume):
        typical_price = np.divide(source, 1)
        typical_price_volume = np.multiply(typical_price, volume)

        cumulative_typical_price_volume = np.cumsum(typical_price_volume)
        cumulative_volume = np.cumsum(volume)
        vwap = np.divide(
            cumulative_typical_price_volume[47:], cumulative_volume[47:])
        vwap = vwap[:-1]
        vwap = np.concatenate([np.full((48,), np.nan), vwap])
        return vwap

    def calculate_vwap(self, source, volume):
        if source is None:
            return None
        typical_prices = np.divide(source, 1)
        typical_prices_volume = typical_prices * volume
        cumulative_typical_prices = np.cumsum(typical_prices_volume)
        cumulative_volumes = np.cumsum(volume)
        vwap = cumulative_typical_prices / cumulative_volumes
        return vwap

    def stoploss(self):
        stop_loss_percent = 0.35
        length_ = 55

        emaC3_ = ta.ema(self.data.close, 50)
        emaC100 = ta.ema(self.data.close, 100)

        highest_low_range = ta.high_low_range(self.data.close, length_)
        return highest_low_range
        # stpLoss= emaC3_[1] > emaC100 and emaC3_ > emaC100  ? lowest : highest

    @staticmethod
    def plot_indicators(self):
        plt.figure(figsize=(10, 6))

        # Plot the indicators
        plt.plot(self.df_trend['ema55H'], label='EMA 55 High', color='blue')
        plt.plot(self.df_trend['ema55L'], label='EMA 55 Low', color='green')
        plt.plot(self.df_trend['ema_100'], label='EMA 100', color='red')
        plt.plot(self.df_trend['ema_200'], label='EMA 200', color='purple')
        plt.plot(self.df_trend['lsma'], label='LSMA', color='orange')
        plt.plot(self.df_trend['ema_10'], label='EMA 10', color='pink')
        plt.plot(self.df_trend['vwap'], label='VWAP', color='brown')
        plt.plot(self.df_trend['wma'], label='WMA', color='gray')
        plt.scatter(self.df_trend['golden_signal'])

        # Add titles and legend
        plt.title('Trebd')
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.legend()

        plt.show()

    def __str__(self):
        return 'trend'


class Screener:
    '''

    SCREENER: be a market maker
    '''

    def __init__(self, data, tradex_logger):
        self.tradex_logger = tradex_logger
        self.data = data
        self.df_screener = pd.DataFrame()
        self.get_screener()

    def get_screener(self) -> pd.DataFrame:
        self.tradex_logger.info('init trade-x screener')
        wma, vwap = self.waves(self.data)

        # moneyflow
        mfi = self.moneyflow()

        # adding the data to the screener DataFrame
        # dots
        dots = self.dots()
        self.df_screener['s_wma'] = wma
        self.df_screener['s_vwap'] = vwap
        self.df_screener['mfi'] = mfi
        self.df_screener['mfi_sum'] = self.mfi_sum
        self.df_screener['s_dots'] = dots

        # adding the data to the general dataframe
        self.data['mfi'] = self.df_screener.mfi

        # self.data['mfi_sum'] =  self.df_screener.mfi_sum
        self.data['s_wma'] = self.df_screener.s_wma
        self.data['s_vwap'] = self.df_screener.s_vwap

        return self.df_screener

    def waves(self, df):
        self.tradex_logger.info('- make the waves')

        df_temp = pd.DataFrame()
        df_temp = df

        n1 = 70
        n2 = 55

        ap = ta.hlc3(df.high, df.low, df.close)
        esa = ta.vwma(ap, df.volume, n1)
        d = ta.ema(abs(ap - esa), n1)
        ci = (ap - esa) / (0.030 * d)
        tci = ta.wma(ci, n2)  # , talib=True
        wt1 = tci
        df_temp['wt2'] = ta.ema(wt1, 4)
        df_temp['s_vwap'] = self.calculate_vwap(df_temp)
        df_temp['s_wma'] = wt1

        return df_temp['s_wma'], df_temp['s_vwap']

    def get_tv_vwap(self, data):
        data['typicalPrice'] = np.divide(data.wt2, 1)
        data['typicalPriceVolume'] = np.multiply(
            data['typicalPrice'], data['volume'])
        cumulative_typical_price_volume = np.cumsum(data['typicalPriceVolume'])
        cumulative_volume = np.cumsum(data['volume'])
        vwap = np.divide(
            cumulative_typical_price_volume[47:], cumulative_volume[47:])
        vwap = vwap[:-1]
        data['vwap'] = np.concatenate([np.full((48,), np.nan), vwap])
        return data['vwap']

    def calculate_vwap(self, data):
        typical_prices = np.divide(data.wt2, 3)
        typical_prices_volume = typical_prices * data['volume']
        cumulative_typical_prices = np.cumsum(typical_prices_volume)
        cumulative_volumes = np.cumsum(data['volume'])
        vwap = cumulative_typical_prices / cumulative_volumes
        return vwap

    def get_vwap2(self, df):
        v = df['volume'].values
        tp = df.wt2
        return df.assign(vwap=(tp * v).cumsum() / v.cumsum())

    def moneyflow(self):
        self.tradex_logger.info('- getting the moneyflow')

        # Moneyflow
        mfi = ta.mfi(self.data['high'], self.data['low'],
                     self.data['close'], self.data['volume'])

        hlc3 = ta.hlc3(self.data['high'], self.data['low'], self.data['close'])

        volume = self.data['volume']

        # Calculate mfi_upper and mfi_lower using rolling sum
        change_hlc3 = np.where(np.diff(hlc3) <= 0, 0, hlc3[:-1])
        mfi_upper = (volume[:-1] * change_hlc3).rolling(window=52).sum()
        mfi_lower = (volume[:-1] * np.where(np.diff(hlc3) >=
                     0, 0, hlc3[:-1])).rolling(window=52).sum()

        # Calculate the Money Flow Index (MFI)
        self.mfi_sum = self._mfi_rsi(mfi_upper, mfi_lower)

        return mfi

    def _mfi_rsi(self, mfi_upper, mfi_lower):
        # Use boolean indexing to compare element-wise
        result = 100.0 - 100.0 / (1.0 + mfi_upper / mfi_lower)

        # Modify the result based on conditions
        result[mfi_lower == 0.1] = 100
        result[mfi_upper == 0.1] = 0

        return result

    def dots(self):
        self.tradex_logger.info('- getting the dots')

        df = self.data

        # gets blue wave and ligth blue wave
        wt1, wt2 = self.waves(df)

        # get the green and red dot

        # 1m
        green_dot = np.where(ta.cross(wt1, wt2), 1, 0)
        dots = np.where(ta.cross(wt2, wt1), -1, green_dot)

        return dots

    def __str__(self):
        return 'screener'


class Real_time:
    '''
    REAL TIME: the fastes way to get market updates

    '''

    def __init__(self, data, tradex_logger):
        self.tradex_logger = tradex_logger
        self.data = data
        self.df_real_time = pd.DataFrame()
        self.get_real_time()

    # TRADE-X SCREENER

    def get_real_time(self) -> pd.DataFrame:

        self.tradex_logger.info('init trade-x Real time')

        # light blue and blue waves
        wt1, wt2 = self.waves(self.data)

        # space between
        space_waves = self.waves_space()

        # dots from multiple time frames
        dots = self.dots()

        # adding the data to the real time DataFrame
        self.df_real_time['b_wave'] = wt2
        self.df_real_time['l_wave'] = wt1
        self.df_real_time['wave_space'] = space_waves

        # dots
        self.df_real_time['dots'] = dots

        # adding the data to the general dataframe
        self.data['b_wave'] = self.df_real_time.b_wave
        self.data['l_wave'] = self.df_real_time.l_wave
        self.data['wave_space'] = self.df_real_time.wave_space

        return self.df_real_time

    def waves(self, data):
        self.tradex_logger.info('- setting up the waves')

        n1 = 10  # channel length
        n2 = 21  # Average Length

        ap = ta.hlc3(data['high'], data['low'], data['close'])
        esa = ta.ema(ap, n1)
        d = ta.ema(abs(ap - esa), n1)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.ema(ci, n2)

        # Light blue Waves
        wt1 = tci
        # blue Waves
        wt2 = ta.sma(wt1, 4)

        return wt1, wt2

    def dots(self):
        self.tradex_logger.info('- getting the dots')

        wt1, wt2 = self.waves(self.data)

        # get the green and red dot

        # 1m
        green_dot = np.where(ta.cross(wt1, wt2), 1, 0)
        dots = np.where(ta.cross(wt2, wt1), -1, green_dot)

        return dots

    def waves_space(self):
        self.tradex_logger.info('- setting up wave spaces')

        # get the waves
        wt1, wt2 = self.waves(self.data)

        # get the space in between the two (when is there a dot comming)
        space_between = wt1 - wt2

        return space_between

    def __str__(self):
        return 'real_time'


class Scanner:
    '''
    SCANNER: scan the market for traps and trends
    '''

    def __init__(self, data, tradex_logger):
        self.tradex_logger = tradex_logger
        self.data = data
        self.df_scanner = pd.DataFrame()
        self.get_scanner()

    def get_scanner(self) -> pd.DataFrame:

        self.tradex_logger.info('init trade-x scanner')
        # the rsi 14 and 40
        rsi14, rsi40 = self.rsis()

        # adding the data to scanner dataframe
        self.df_scanner['rsi14'] = rsi14
        self.df_scanner['rsi40'] = rsi40

        # adding the data to the general dataframe
        self.data['rsi14'] = rsi14
        self.data['rsi40'] = rsi40

        return self.df_scanner

    def rsis(self):
        self.tradex_logger.info("- setting up the rsi's")

        # make the rsi's
        rsi14 = ta.rsi(self.data['close'], 14)
        rsi40 = ta.rsi(self.data['close'], 40)
        # rsi14.fillna(inplace=True, value=0)
        # rsi40.fillna(inplace=True, value=0)

        return rsi14, rsi40

    def divergences(self):
        pass

    def __str__(self):
        return 'scanner'


'''
def create_signals(self) -> None:
        
        self.tradex_logger.info('creating the signals from trade-x')
        
        df10 = self.changeTime(self.data, '10min')
        df30 = self.changeTime(self.data, '30min')
        df60 = self.changeTime(self.data, '60min')
        
   
        ##########   TREND   #########
        
        #Getting the Signals for the 55 high/low channel
        ema55H, ema55L = self.trend.df_trend.ema55H, self.trend.df_trend.ema55L
        ema_upper = np.where(self.data['close'] > ema55H, 1, 0)
        ema_channel = np.where(self.data['close'] < ema55L, -1, ema_upper)
        
        #Getting the signal for 200 ema trend
        ema_200, ema_100 = self.trend.df_trend.ema_100, self.trend.df_trend.ema_200
        ema_trend = np.where(ema_100 > ema_200, 1, -1)
        ema_cross = np.where(ta.cross(ema_100, ema_200), 1, 0)
        ema_cross = np.where(ta.cross(ema_200, ema_100), -1, ema_cross)
        
        #Getting the signal for lsma trend
        lsma, ema_10 = self.trend.df_trend.lsma, self.trend.df_trend.ema_10
        lsma_cross_ema1 = np.where(ta.cross(ema_10, lsma), 1, 0)
        lsma_cross_ema = np.where(ta.cross(lsma, ema_10), -1, lsma_cross_ema1)
        
        #Getting the signal for Vwap trend
        vwap, wma = self.trend.df_trend.vwap, self.trend.df_trend.wma
        vwap_cross1 = np.where(ta.cross(wma, vwap), 1, 0)
        vwap_cross = np.where(ta.cross(vwap, wma), -1, vwap_cross1)
        vwap_buy_sell_signal1 = np.where(vwap_cross & ema_channel == 1, 1, 0)
        vwap_buy_sell_signal = np.where(
        vwap_cross & ema_channel == -1, -1, vwap_buy_sell_signal1)
        
        ######### SCREENER ###########
        
        wma10, vwap10 = self.screener.waves(df10)
        wma30,vwap30 = self.screener.waves(df30)
        wma60, vwap60 = self.screener.waves(df60)
        
        #10min
        s_green_dot10 = np.where(ta.cross(wma10, vwap10), 1, 0)
        s_dots10 = np.where(ta.cross(vwap10, wma10), -1, s_green_dot10)
        
        #30min
        s_green_dot30 = np.where(ta.cross(wma30, vwap30), 1, 0)
        s_dots30 = np.where(ta.cross(vwap30, wma30), -1, s_green_dot30)
        
        #60min
        s_green_dot60 = np.where(ta.cross(wma60, vwap60), 1, 0)
        s_dots60 = np.where(ta.cross(vwap60, wma60), -1, s_green_dot60)
        s_dots60_trend = np.where(wma60 > vwap60, 1, -1)
        
        # self.df_screener['dots_10'] = pd.DataFrame(dots10).fillna(0)
        # self.df_screener['dots_30'] = pd.DataFrame(dots30).fillna(0)
        # self.df_screener['dots_60'] = pd.DataFrame(dots60).fillna(0)
        # self.df_screener['dots60_trend'] = pd.DataFrame(dots60_trend).fillna(0)
        

        ######### REAL TIME ###########
        
        #gets blue wave and ligth blue wave
        wt11, wt21 = self.real_time.waves(df10)
        wt13, wt23 = self.real_time.waves(df30)
        wt16, wt26 = self.real_time.waves(df60)
        
        #10min
        green_dot10 = np.where(ta.cross(wt11, wt21), 1, 0)
        dots10 = np.where(ta.cross(wt21, wt11), -1, green_dot10)
        
        #30min
        green_dot30 = np.where(ta.cross(wt13, wt23), 1, 0)
        dots30 = np.where(ta.cross(wt23, wt13), -1, green_dot30)
        
        #60min
        green_dot60 = np.where(ta.cross(wt16, wt26), 1, 0)
        dots60 = np.where(ta.cross(wt26, wt16), -1, green_dot60)
        dots60_trend = np.where(wt16 > wt26, 1, -1)
        
        #########   SCANNER   #########
        
        #Get the RSI SIGNALS
        rsi14, rsi40 = self.scanner.df_scanner.rsi14, self.scanner.df_scanner.rsi40
        rsi14p = rsi14.shift(1)
        rsi40p = rsi40.shift(1)
        rsi14.fillna(0, inplace=True)
        rsi14p.fillna(0, inplace=True)
        rsi40.fillna(0, inplace=True)
        rsi40p.fillna(0, inplace=True)
        rsi_trend1 = np.where(rsi40 > 46.5, -1, 0)
        rsi_trend2 = np.where(rsi40 < 49.82, -1, rsi_trend1)
        rsi_trend = np.where(rsi40 > 49.82, 1, rsi_trend2)
        
        #Get the RSI SIGNALS FOR OVERBOUGHT AND OVERSOLD
        def find_crossover(rsi, rsip):
            if rsi > 30 and rsip < 30:
                return 1
            elif rsi < 70 and rsip > 70:
                return -1
            return 0
        def find_crossover40(rsi, rsip):
            if rsi > 49.2 and rsip < 49.2:
                return 1
            elif rsi < 49.2 and rsip > 49.2:
                return -1
            return 0
        rsi_overbought = np.where(rsi14 > 70, -1, 0)
        rsi_sb = np.where(rsi14 < 28, 1, rsi_overbought)
        
        #### SIGNALS INTO DATAFRAME #####
        
        #Trend Dataframe
        self.trend.df_trend['ema_channel'] = ema_channel
        self.trend.df_trend['lsma_cross_ema'] = lsma_cross_ema
        self.trend.df_trend['vwap_cross'] = vwap_cross
        self.trend.df_trend['vwap_buy_sell'] = vwap_buy_sell_signal
        self.trend.df_trend['ema_trend'] = ema_trend
        
        #Screener Dataframe
        self.screener.df_screener['s_dots10'] = pd.Series(s_dots10)
        self.screener.df_screener['s_dots30'] = pd.Series(s_dots30)
        self.screener.df_screener['s_dots60'] = pd.Series(s_dots60)
        self.screener.df_screener['s_dots60_trend'] = pd.Series(s_dots60_trend)
        
    
        #Real Time Dataframe
        self.real_time.df_real_time['dots10'] = pd.Series(dots10).fillna(inplace=True, value=0)
        self.real_time.df_real_time['dots30'] = pd.Series(dots30).fillna(inplace=True, value=0)
        self.real_time.df_real_time['dots60'] = pd.Series(dots60).fillna(inplace=True, value=0)
        self.real_time.df_real_time['dots60_trend'] = pd.Series(dots60_trend).fillna(inplace=True, value=0)
        
        
        
        
        #Scanner Dataframe
        self.scanner.df_scanner['rsi_trend'] = rsi_trend
        self.scanner.df_scanner['rsi_sb'] = rsi_sb
        self.scanner.df_scanner['rsi_b_s14'] = np.vectorize(find_crossover)(rsi14, rsi14p)
        self.scanner.df_scanner['rsi40_buy_sell'] = np.vectorize(find_crossover40)(rsi40, rsi40p)

        #general Dataframe
        
        
        #T
    
        #S
    
        #RT
        
        #SC
    

'''
