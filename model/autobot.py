import time


class AutoBot:
    def __init__(self,exchange, symbol, amount, stop_loss, take_profit, model, time, ml, trade_x,logger):
        # Initialize class variables as before
        self.exchange = exchange
        self.symbol = symbol
        self.amount = amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.model = model 
        self.time = time
        self.ml = ml
        self.trade_x = trade_x
        self.open_orders = {}
        
        
        self.auto_trade = False
        # Set up logging
        self.logger = logger
        
    def getnormal_symbol(self):
        symbol = self.symbol.replace('/','').lower()
        return symbol
    
    def getnormal_count(self):
        if self.time == "1m":
            return 60
        if self.time == "5m":
            return 300
        if self.time == "30m":
            return 1_800
        if self.time == "1h":
            return 3_600
        if self.time == "3h":
            return 10_800
        
    def start_auto_trading(self, event):
        self.auto_trade = event
        self._run_auto_trading()


            
    def stop_auto_trading(self, event):
        self.auto_trade = event
       
        
    def _run_auto_trading(self):
        while self.auto_trade:
            if self.auto_trade == False:
                break
            # Get the current price of the asset
            current_price = self.exchange.fetch_ticker(self.symbol)['last']
            
            # Use the model to predict the next price
            prediction = self.ml.predict(self.model, self.time, self.symbol)
            print(prediction)
            
            #Use the trade_x to get the signal
            trade_x_signal = self.trade_x.trade_x()
            print(trade_x_signal)
            
            # Check if the prediction is above the take profit or below the stop loss
            if prediction == 2:
                
                open_orders = self.exchange.fetch_open_orders(symbol=self.symbol)
                for order in open_orders:
                    if order['side'] == 'sell':
                        self.logger.info(f'sell order for {self.symbol} already exists with id: {order["id"]}')
                        if self.check_order_status(order['id']):
                            self.open_orders.pop(order['id'], None)
                        else:
                            return
                # Place a sell order
                try:
                    order = self.exchange.create_order(symbol=self.symbol, type='limit', side='sell', amount=self.amount, price=current_price)
                    self.logger.info(f'Sold {self.amount} {self.symbol} at {current_price}')
                    stop_loss_order = self.exchange.create_order(symbol=self.symbol, type='stoploss', side='sell', amount=self.amount, price=current_price / (1 + self.stop_loss / 100))
                    self.open_orders[order['id']] = {'symbol': self.symbol, 'side': 'sell','amount': self.amount, 'price': current_price}
                except Exception as e:
                    self.logger.error(e)
                    
            elif prediction == 1:
                
                open_orders = self.exchange.fetch_open_orders(symbol=self.symbol)
                for order in open_orders:
                    if order['side'] == 'buy':
                        self.logger.info(f'buy order for {self.symbol} already exists with id: {order["id"]}')
                        if self.check_order_status(order['id']):
                            self.open_orders.pop(order['id'], None)
                        else:
                            return
                    
                # Place a buy order
                try:
                    order = self.exchange.create_order(symbol=self.symbol, type='limit', side='buy', amount=self.amount, price=current_price) 
                    self.logger.info(f'Bought {self.amount} {self.symbol} at {current_price}')
                    stop_loss_order = self.exchange.create_order(symbol=self.symbol, type='stoploss', side='sell', amount=self.amount, price=current_price * (1 + self.stop_loss / 100) )          
                    self.open_orders[order['id']] = {'symbol': self.symbol, 'side': 'buy','amount': self.amount, 'price': current_price}
                except Exception as e:
                    self.logger.error(e)
            elif prediction == 0:
                    self.logger.info(f"the prediction gave {prediction} waiting for a trading oppertunity!")
            time.sleep(self.getnormal_count())
            
    def check_order_status(self, order_id):
        order = self.exchange.fetch_order(order_id, self.symbol)
        if order['status'] == 'closed' or order['status'] == 'filled':
            return True
        else:
            return False

    def __str__(self):
        return f"Autobot-{self.exchange}: {self.symbol}-{self.time}-{self.model}- Trades: {len(self.open_orders)} -> "