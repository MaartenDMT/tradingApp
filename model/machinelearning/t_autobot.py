import asyncio
import json
import time

import nest_asyncio
import pandas as pd
import requests
import websockets

nest_asyncio.apply()


class AutoBot:
    def __init__(self, exchange, symbol, amount, stop_loss, take_profit, model, time, ml, logger) -> None:
        self.exchange = exchange
        self.symbol = symbol.replace('/', '')
        self.amount = amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.model = model
        self.time = time
        self.ml = ml

        self.auto_trade = False
        # Set up logging
        self.logger = logger

        self.df = self.get_data()

    def predict(self):
        pred = self.ml.predict(self.model, self.time)
        return pred

    def get_data(self) -> pd.DataFrame:
        # Retrieve historical data from the Binance API
        response = requests.get(
            f'https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval={self.time}')
        historicalData = response.json()

        # Convert historical data to a pandas DataFrame
        df = pd.DataFrame({'date': historicalData[0],
                           'open': historicalData[1],
                           'high': historicalData[2],
                           'low': historicalData[3],
                           'close': historicalData[4],
                           'volume': historicalData[5]})

        df.date = pd.to_datetime(df.date, unit='ms')

        # Use pandas and NumPy to analyze and manipulate the data
        self.mean = df['close'].astype(float).mean()
        self.std = df['close'].astype(float).std()

        return df

    # Set up WebSocket connection using the websockets library
    async def receive_data(self):
        async with websockets.connect(f'wss://stream.binance.com/ws/{self.symbol.lower()}@kline_{self.time}') as websocket:
            while True:
                data = await websocket.recv()
                data = json.loads(data)
                candle = data['k']
                # Convert data to a pandas DataFrame

                self.df = pd.DataFrame({'date': candle['t'], 'open': candle['o'], 'high': candle['h'],
                                       'low': candle['l'], 'close': candle['c'], 'volume': candle['v']}, index=[0])
                self.df.date = pd.to_datetime(self.df.date, unit='ms')

                # Use the data to make decisions about when to buy and sell
                current_price = self.df['close'].astype(float).iloc[0]
                if current_price > self.mean + self.std:
                    # Place a sell order
                    try:
                        self.exchange.create_order(
                            symbol=self.symbol, type='limit', side='sell', amount=self.amount, price=current_price)
                        self.logger.info(
                            f'Sold {self.amount} {self.symbol} at {current_price}')
                        stop_loss_order = self.exchange.create_order(
                            symbol=self.symbol, type='stop_loss', side='sell', amount=self.amount, price=self.stop_loss)
                    except Exception as e:
                        self.logger.error(e)
                elif current_price < self.mean + self.std:
                    # Place a buy order
                    try:
                        self.exchange.create_order(
                            symbol=self.symbol, type='limit', side='buy', amount=self.amount, price=current_price)
                        self.logger.info(
                            f'Buy {self.amount} {self.symbol} at {current_price}')
                        stop_loss_order = self.exchange.create_order(
                            symbol=self.symbol, type='stop_loss', side='sell', amount=self.amount, price=self.stop_loss)
                    except Exception as e:
                        self.logger.error(e)
                time.sleep(1_800)

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.receive_data())
