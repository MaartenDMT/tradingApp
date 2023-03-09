from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Trading:
    def __init__(self, exchange, symbol, position_size, max_drawdown,
                 moving_average_period, decision_tree_depth, logger) -> None:

        self.exchange = exchange 
        self.getBalancUSDT, self.getBalancBTC = self.getBalance()
        self.logger = logger
        self.symbol = symbol
        self.stoploss = 0
        self.takeprofit1 = 0
        self.takeprofit2 = 0
        self.takeprofit3 = 0
        self.auto_trade = False
        self.in_trade = False
        self.trade_id = None
        self.trailing_stop = False
        self.position_size = position_size
        self.max_drawdown = max_drawdown
        self.moving_average_period = moving_average_period
        self.decision_tree_depth = decision_tree_depth
        self.analyzer = SentimentIntensityAnalyzer()

        
    def check_trade_status(self) -> bool:
        if not self.in_trade:
            return False
        else:
            try:
                # Check the status of the trade on the exchange
                trade = self.exchange.fetch_order(self.trade_id)
                self.logger(f'Checking the status of trade with ID {trade}')
                if trade['status'] == 'closed':
                    self.in_trade = False
                    self.trade_id = None
                    return False
                else:
                    return True
            except Exception as e:
                self.logger.error(f"in the check_trade_status Error: {e}")

    def place_trade(self, side, trade_type, amount, price, takeprofit, stoploss) -> None:

        # Calculate the stop loss and take profit levels based on the current price and the maximum variance
        self.stoploss = price - stoploss if self.side == 'buy' else price + stoploss
        self.takeprofit1 = price + takeprofit if self.side == 'buy' else price - takeprofit
        self.takeprofit2 = price + (takeprofit + (takeprofit * 0.4)
                                    ) if self.side == 'buy' else price - (takeprofit + (takeprofit * 0.4))
        self.takeprofit3 = price + \
            (takeprofit+(takeprofit * 0.8)) if self.side == 'buy' else price - \
            (takeprofit + (takeprofit * 0.8))

        # Place a trade on the exchange with the specified parameters
        try:
            order = self.exchange.create_order(symbol=self.symbol, type=trade_type, side=side, amount=amount, price=price, params={
                'stopLoss': self.stoploss,
                'takeProfit': self.takeprofit1})
            
            self.logger.info(f"{self.symbol} {side} Order Placed with order Type: +\
                {trade_type} {amount} at {price} with a stop at {self.stoploss}+\
                    and take profit {self.takeprofit1} | {self.takeprofit2} | {self.takeprofit2} ")
            self.in_trade = True
            self.trade_id = order['id']
        except Exception as e:
            self.logger.error(f"in the place_trade Error: {e}")

    def scale_in_out(self, amount) -> None:
        # Check if we are in a trade
        if self.check_trade_status():
            # If we are, update the size of the trade
            self.exchange.update_order(self.trade_id, {'amount': amount})

    def check_and_place_trade(self, trade_type, amount, price, takeprofit, stoploss, predict) -> None:

        # Check if we are already in a trade
        if not self.check_trade_status():

            # Check if the trade meets the risk management criteria
            if amount * price < self.position_size * self.exchange.fetch_balance()['USDT']['total'] and + \
                    self.exchange.fetch_ticker(self.symbol)['bid'] > self.stoploss * (1 - self.max_drawdown):

                # Make a prediction using the model
                prediction = predict

                # Check if the prediction is 1 (buy) or -1 (sell)
                if prediction == 1:

                    # If it is 1, place a buy trade
                    try:
                        self.place_trade('buy', trade_type,
                                         amount, price, takeprofit, stoploss)
                        self.logger.info(
                            f"Placing Buying order Type: {trade_type} {amount} at {price}")
                    except Exception as e:
                        self.logger.error(
                            f"in the check_and_place_trade (buy) Error: {e}")
                elif prediction == -1:
                    # If it is -1, place a sell trade
                    try:
                        self.place_trade(
                            'sell', trade_type, amount, price, takeprofit, stoploss)
                        self.logger.info(
                            f"Placing Selling order Type: {trade_type} {amount} at {price}")
                    except Exception as e:
                        self.logger.error(
                            f"in the check_and_place_trade (sell) Error: {e}")

    def check_and_modify_trade(self, price) -> None:

        # Check if we are in a trade
        if self.check_trade_status():

            # If we are, check if we need to modify the stop loss or take profit levels
            trade = self.exchange.fetch_order(self.trade_id)
            if self.side == 'buy':

                # For a long trade, check if the current price is above the take profit levels
                if price >= self.takeprofit1:
                    self.exchange.update_order(
                        self.trade_id, {'takeProfit': self.takeprofit2})
                if price >= self.takeprofit2:
                    self.exchange.update_order(
                        self.trade_id, {'takeProfit': self.takeprofit3})
                if self.trailing_stop:

                    # If we are using a trailing stop, update the stop loss to the current price minus the trailing amount
                    self.exchange.update_order(
                        self.trade_id, {'stopLoss': price - self.trailing_stop})
            else:

                # For a short trade, check if the current price is below the take profit levels
                if price <= self.takeprofit1:
                    self.exchange.update_order(
                        self.trade_id, {'takeProfit': self.takeprofit2})
                if price <= self.takeprofit2:
                    self.exchange.update_order(
                        self.trade_id, {'takeProfit': self.takeprofit3})
                if self.trailing_stop:

                    # If we are using a trailing stop, update the stop loss to the current price minus the trailing amount
                    self.exchange.update_order(
                        self.trade_id, {'stopLoss': price + self.trailing_stop})

    def getBalance(self):
        
        usdt = self.exchange.fetch_balance()['USDT']['total']
        btc = self.exchange.fetch_balance()['BTC']['total']

        return usdt, btc
    
    def getBidAsk(self):
        
        #get orderbook the bid/ask prices
        orderbook = self.exchange.fetch_ticker(self.symbol)
        
        bid = orderbook['bid']
        ask = orderbook['ask']
        
        return bid, ask

    def scale_in_out(self, amount) -> None:
        # Check if we are in a trade
        if self.check_trade_status():
            # If we are, update the size of the trade
            self.exchange.update_order(self.trade_id, {'amount': amount})
