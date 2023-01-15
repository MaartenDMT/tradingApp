import time

import numpy as np
import pandas as pd
import pytab as pt
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


class NeuralNetwork:
    def __init__(self):
        pass

    def start_bot(self):
        
        # Set the stop flag to False to indicate that the bot is running
        self.stop_flag = False
        
        # Get the current balance on each exchange
        balances = [exchange.fetch_balance() for exchange in self.exchanges]
       
        # Calculate the total balance across all exchanges
        total_balance = sum([balance['total']['BTC'] for balance in balances])
        
        # Set the trade amount as a percentage of the total balance
        trade_amount = total_balance * 0.1
        
        # Start the trading loop
        while not self.stop_flag:
            
            # Get the current market data for each exchange
            markets = [exchange.fetch_ticker('BTC/USDT') for exchange in self.exchanges]
            
            # Convert the market data to a Pandas DataFrame
            df = pd.DataFrame(markets)
            
            # Preprocess the data
            df = self.preprocess_data(df)
           
            # Use the model to make a prediction
            prediction = self.predict(df)
            
            # Place a buy or sell order on each exchange based on the prediction
            for exchange, p in zip(self.exchanges, prediction):
                if p > 0:
                    # exchange.create_market_buy_order('BTC/USDT', trade_amount)
                    self.place_trade(exchange, 'BTC/USDT', 'buy', trade_amount)
                else:
                    # exchange.create_market_sell_order('BTC/USDT', trade_amount)
                    self.place_trade(exchange, 'BTC/USDT', 'sell', trade_amount)
                    
            # Sleep for 60 seconds before making the next prediction
            time.sleep(60)
            
    def stop_bot(self):
                    
        # Set a flag to indicate that the bot should stop running
        self.stop_flag = True

        # Update the status label
        self.status_label.config(text='Status: Stopped')
        
    def view_trade_history(self):
        
        # Create a new window
        trade_history_window = tk.Toplevel(self.parent)
        trade_history_window.title('Trade History')
        
        # Convert the trade history data to a Pandas DataFrame
        df = pd.DataFrame(self.trade_history)
        
        # Create a table widget and add it to the window
        table = pt.table(trade_history_window, dataframe=df, showtoolbar=True, showstatusbar=True)
        table.show()
        
    def load_model(self):
        
        # Load the model from a file
        model = tf.keras.models.load_model('data/model/model.h5')
        return model
    
    def train_model(self, data):
        
        # Preprocess the data
        data = self.preprocess_data(data)
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)
        
        # Build the model
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

        # Save the model to a file
        model.save('data/model/model.h5')
        return model

    def preprocess_data(self, data):
        
        # Calculate the midpoint price for each exchange
        data['midpoint'] = (data['bid'] + data['ask']) / 2
        
        # Calculate the difference between the midpoint prices of the exchanges
        data['diff'] = data['midpoint'] - data['midpoint'].shift(1)
        
        # Drop any rows with missing values
        data.dropna(inplace=True)
        
        # Calculate the target column
        data['target'] = np.where(data['diff'] > 0, 1, 0)
        
        # Drop unnecessary columns
        data.drop(['timestamp', 'bid', 'ask'], axis=1, inplace=True)
        return data
    
    def predict(self, data):
        
        # Use the model to make a prediction on the data
        prediction = self.model.predict(data)
        
        # Round the prediction to the nearest integer
        prediction = np.round(prediction)
        return prediction
    
    def place_trade(self, exchange, symbol, side, amount):
        
        # Connect to the exchange
        client = exchange #ccxt.binance()
        
        # Check if the exchange is enabled
        if exchange not in self.enabled_exchanges:
            return
        
        # Check if the exchange is authenticated
        if not client.apiKey:
            return
        
        # Place the trade
        try:
            client.create_order(symbol, 'market', side, amount)
        except Exception as e:
            print(e)
            
    def update_balance(self, exchange):
        
        # Connect to the exchange
        client = exchange
        
        # Check if the exchange is enabled
        if exchange not in self.enabled_exchanges:
            return
        
        # Check if the exchange is authenticated
        if not client.apiKey:
            return
        
        # Update the balance data
        try:
            balance = client.fetch_balance()
        except Exception as e:
            print(e)
        
        # Update the balance data for the exchange
        self.balances[exchange] = balance
    
