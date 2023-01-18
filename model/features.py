import pandas_ta as ta
import numpy as np
import pandas as pd
import vectorbt as vbt

class Tradex_indicator:

    def __init__(self,symbol,t:None, get_data:False, data):
        self.symbol = symbol 
        self.data = self.changeTime(self.get_data(get_data, data), t)
        print(self.data)
        self.trend = Trend(self.data)
        self.screener = Screener(self.data)
        self.real_time = Real_time(self.data)
        self.scanner = Scanner(self.data)
       
        
        
    
    def get_data(self, get_data, data) -> pd.DataFrame:
        if get_data:
            print('getting the data')
            data_load=vbt.CCXTData.download([self.symbol], start='3 days ago', timeframe='1m')
            df  = data_load.get()
            df = pd.DataFrame(df)
            # Convert the Open time column to a datetime object
            df['date'] = pd.to_datetime(df.index)

            # Set the Open time column as the index of the DataFrame
            df.set_index('date', inplace=True)
            df['symbol'] = self.symbol
            # Rename the columns to match the desired column names
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            result = pd.concat([data,df])
            result.dropna(inplace=True)
            return result
        
        else:
            return data

    def changeTime(self, data:pd.DataFrame, t) -> pd.DataFrame:
        data.index = pd.to_datetime(data.index, utc=True)
        if t != None:
            print(f'getting the right timeframe {t}')
            ohlc = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df = data.resample(t, label='left', kind='timestamp').apply(ohlc)
            df.dropna(inplace=True)

            #df=df.iloc[100:200]
            df = df.reset_index()
            df.index = pd.to_datetime(df.index)
            df.set_index(['date'], inplace=True)
        else:
            df = data

        return df

# make the features of the trade-x screener and the trade-x trend

    def trade_x(self):
        self.create_signals()
        
        return self.data


    def create_signals(self):
        
        print('creating the signals from trade-x')
        
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
    TREND: visueel beeld van de market trend
'''
class Trend:
    def __init__(self, data):
        self.data = data
        self.df_trend = pd.DataFrame()
        self.get_trend()
        
    # TRADE - X TREND
    
    def get_trend(self):
        print('init trade-x trend')
        
        #EMA channel 
        ema55H, ema55L = self.hlChannel()
        
        #EMA trend lines
        ema_200, ema_100 = self.ema()
        
        
        #lsma and ema 
        lsma, ema_10 = self.lsma_()
        
        #vwap and wma
        vwap, wma = self.vwap_()
        
        #adding the data to the trend dataframe
        
        self.df_trend['ema55H'] = ema55H
        self.df_trend['ema55L'] = ema55L
        self.df_trend['ema_100'] = ema_100
        self.df_trend['ema_200'] = ema_200
        self.df_trend['lsma'] = lsma
        self.df_trend['ema_10'] = ema_10
        self.df_trend['vwap'] = vwap
        self.df_trend['wma'] = wma
    
        #adding the data to the general dataframe
        self.data['ema55H'] = self.df_trend.ema55H
        self.data['ema55L'] = self.df_trend.ema55L
        self.data['ema_100'] =self.df_trend.ema_100
        self.data['ema_200'] = self.df_trend.ema_200
        self.data['lsma'] = self.df_trend.lsma
        self.data['ema_10'] = self.df_trend.ema_10
        # self.data['vwap'] = trend.vwap
        # self.data['wma'] = trend.wma
        
        return self.df_trend

    def hlChannel(self):
        print('- setting up High|Low channel')

        #create the 55 ema channel

        ema55H = ta.ema(self.data['high'], 55)
        ema55L = ta.ema(self.data['low'], 55)

        return ema55H, ema55L


    def ema(self):
        print('- setting up EMA')
        

        # create the ema trend
        ema_200 = ta.ema(self.data['close'], 200)
        ema_100 = ta.ema(self.data['close'], 100)

        return ema_200, ema_100

    def lsma_(self):
        print('- setting up LSMA')
        
        lsma = ta.linreg(self.data['low'], 20, 8)
        ema_10 = ta.ema(self.data['close'], 10)

        return lsma, ema_10


    def vwap_(self):
        print('- setting up VWAP')
        
        vwma = ta.vwma(self.data['close'], self.data['volume'], 14)
        wma = ta.wma(vwma, 21)
        vwap = self.get_tv_vwap(wma)
        # Calculate the VWAP using the "price" and "volume" columns
        # vwap = (df["price"] * df["volume"]).sum() / df["volume"].sum()

        return vwap, wma

    def get_tv_vwap(self,source):
        data = self.data
        data['source'] = source
        data['typicalPrice'] = (data.source).div(1).values
        data['typicalPriceVolume'] = data['typicalPrice'] * data['volume']
        data['cumulativeTypicalPriceVolume1'] = data['typicalPriceVolume'].rolling(48).sum()
        data['cumulativeVolume1'] = data['volume'].rolling(48).sum()
        data['vwap'] = data['cumulativeTypicalPriceVolume1']/data['cumulativeVolume1']
        return data.vwap  


'''
    SCREENER: be a market maker
'''

class Screener:
    def __init__(self, data):
        self.data = data
        self.df_screener = pd.DataFrame()
        self.get_screener()
        

    def get_screener(self):
        print('init trade-x screener')
        wma, vwap = self.waves(self.data)
        #moneyflow
        mfi = self.moneyflow()
        #adding the data to the screener DataFrame
        #dots
        dots = self.dots()
        self.df_screener['s_wma'] = wma
        self.df_screener['s_vwap'] = vwap
        self.df_screener['dots'] = dots
        
        self.df_screener['mfi'] = mfi
        # self.df_screener['mfi_sum'] = pd.Series(self.mfi_sum)
        
        
        #adding the data to the general dataframe
        self.data['mfi'] = self.df_screener.mfi
        # self.data['mfi_sum'] =  self.df_screener.mfi_sum
        self.data['s_wma'] = self.df_screener.s_wma
        self.data['s_vwap'] = self.df_screener.s_vwap
        
        return self.df_screener
    
    def waves(self, df):
        print('- make the waves')
       
        n1 = 70 
        n2 = 55

        ap = ta.hlc3(df.high, df.low, df.close)
        esa = ta.vwma(ap, df.volume, n1)
        print(esa)
        d = ta.ema(abs(ap - esa), n1)
        ci = (ap - esa) / (0.030 * d)
        tci = ta.wma(ci, n2) #, talib=True
        wt1 = tci
        v = ta.ema(wt1, 4)

        df_temp = pd.DataFrame()
        df_temp = df
        df_temp['wt2'] = v
        df_temp['volume'] = df.volume
        df_temp['s_vwap'] = self.get_tv_vwap(df_temp)
        df_temp['s_wma'] = wt1
        
        return df_temp['s_wma'] ,df_temp['s_vwap']
    
    def get_tv_vwap(self,data):
      
        
        data['typicalPrice'] = (data.wt2).div(1).values
        data['typicalPriceVolume'] = data['typicalPrice'] * data['volume']
        data['cumulativeTypicalPriceVolume1'] = data['typicalPriceVolume'].rolling(48).sum()
        data['cumulativeVolume1'] = data['volume'].rolling(48).sum()
        data['vwap'] = data['cumulativeTypicalPriceVolume1']/data['cumulativeVolume1']
        return data.vwap  
    
    def get_vwap2(self,df):
        v = df['volume'].values
        tp = df.wt2
        return df.assign(vwap=(tp * v).cumsum() / v.cumsum())
        
    def moneyflow(self):
        print('- getting the moneyflow')
        # Moneyflow
        mfi = ta.mfi(self.data['high'], self.data['low'], self.data['close'], self.data['volume'])

        #Money flow
        ap = ta.hlc3(self.data['high'], self.data['low'], self.data['close'])
        typical_price = ap
        period = 14

        money_flow = typical_price * self.data['volume']

        #Get all of the positive and negative money flows
        #where the current typical price is higher than the previous day's typical price, we will append that days money flow to a positive list
        #and where the current typical price is lower than the previous day's typical price, we will append that days money flow to a negative list
        #and set any other value to 0 to be used when summing

        positive_flow = []  # Create a empty list called positive flow
        negative_flow = []  # Create a empty list called negative flow

        #Loop through the typical price
        for i in np.arange(1, len(typical_price)):

            # if the present typical price is greater than yesterdays typical price
            if typical_price[i] > typical_price[i-1]:

                # Then append money flow at position i-1 to the positive flow list
                positive_flow.append(money_flow[i-1])
                negative_flow.append(0)  # Append 0 to the negative flow list

                # if the present typical price is less than yesterdays typical price
            elif typical_price[i] < typical_price[i-1]:

                # Then append money flow at position i-1 to negative flow list
                negative_flow.append(money_flow[i-1])
                positive_flow.append(0)  # Append 0 to the positive flow list

            else:

                # Append 0 if the present typical price is equal to yesterdays typical price
                positive_flow.append(0)
                negative_flow.append(0)

        #Get all of the positive and negative money flows within the time period
        positive_mf = []
        negative_mf = []

        #Get all of the positive money flows within the time period
        
        for i in np.arange(period-1, len(positive_flow)):

            positive_mf.append(np.sum(positive_flow[i+1-period: i+1]))

        #Get all of the negative money flows within the time period
        for i in np.arange(period-1, len(negative_flow)):

            negative_mf.append(np.sum(negative_flow[i+1-period: i+1]))

        self.mfi_sum = 100 * (np.array(positive_mf) /
                    (np.array(positive_mf) + np.array(negative_mf)))

        return mfi
    
    def dots(self):
        print('- getting the dots')
        

        df = self.data
        
        #gets blue wave and ligth blue wave
        wt1, wt2 = self.waves(df)
       
        #get the green and red dot

        #1m
        green_dot = np.where(ta.cross(wt1, wt2), 1, 0)
        dots = np.where(ta.cross(wt2, wt1), -1, green_dot)


        return dots

'''
    REAL TIME: the fastes way to get market updates

'''
class Real_time:
    
    def __init__(self, data):
        self.data = data
        self.df_real_time = pd.DataFrame()
        self.get_real_time()
    
    # TRADE-X SCREENER
    def get_real_time(self):
        
        print('init trade-x Real time')
        
        # light blue and blue waves
        wt1, wt2 = self.waves(self.data)
        
        #space between
        space_waves = self.waves_space()
        
        #dots from multiple time frames
        dots = self.dots()
        
        #adding the data to the real time DataFrame
        self.df_real_time['b_wave'] = wt2
        self.df_real_time['l_wave'] = wt1
        self.df_real_time['wave_space'] = space_waves
        
        #dots
        self.df_real_time['dots'] = dots
        
        #adding the data to the general dataframe
        self.data['b_wave'] = self.df_real_time.b_wave
        self.data['l_wave'] = self.df_real_time.l_wave
        self.data['wave_space'] = self.df_real_time.wave_space
        
        return self.df_real_time
      

    def waves(self, data):
        print('- setting up the waves')
        
        n1 = 10  # channel length
        n2 = 21  # Average Length

        ap = ta.hlc3(data['high'], data['low'], data['close'])
        esa = ta.ema(ap, n1)
        d = ta.ema(abs(ap - esa), n1)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.ema(ci, n2)


        #Light blue Waves
        wt1 = tci
        #blue Waves
        wt2 = ta.sma(wt1, 4)

        return wt1, wt2
    
    def dots(self):
        print('- getting the dots')
        
        wt1, wt2 = self.waves(self.data)

        #get the green and red dot

        #1m
        green_dot = np.where(ta.cross(wt1, wt2), 1, 0)
        dots = np.where(ta.cross(wt2, wt1), -1, green_dot)

        return dots
    
    def waves_space(self):
        print('- setting up wave spaces')
        
        #get the waves
        wt1,wt2 = self.waves(self.data)

        #get the space in between the two (when is there a dot comming)
        space_between = wt1 - wt2

        return space_between

'''
    SCANNER: scan the market for traps and trends
'''
    
class Scanner:
    
    def __init__(self, data):
        self.data = data
        self.df_scanner = pd.DataFrame()
        self.get_scanner()
        
    def get_scanner(self):
        
        print('init trade-x scanner')
        #the rsi 14 and 40
        rsi14, rsi40 = self.rsis()
        
        #adding the data to scanner dataframe
        self.df_scanner['rsi14'] = rsi14
        self.df_scanner['rsi40']= rsi40
        
        
        #adding the data to the general dataframe
        self.data['rsi14'] = self.df_scanner.rsi14
        self.data['rsi40'] = self.df_scanner.rsi40
        
        return self.df_scanner
    
    def rsis(self):
        print("- setting up the rsi's")
        
        #make the rsi's
        rsi14 = ta.rsi(self.data['close'], 14)
        rsi40 = ta.rsi(self.data['close'], 40)
        rsi14.fillna(inplace=True, value=0)
        rsi40.fillna(inplace=True, value=0)

        return rsi14, rsi40

    def divergences(self):
        pass


# trade_x = Tradex_indicator('BTCUSDT', '30m')

# print(trade_x)