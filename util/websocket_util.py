import json
import threading

import websocket


class WebSocketClient:
    def __init__(self, symbol, on_message):
        self.symbol = self.adjust_string(symbol)
        self.on_message = on_message
        self.ws = None

    @staticmethod
    def adjust_string(s=str) -> str:
        s = s.lower()
        s = s.replace("/", "")
        return s

    def _on_message(self, ws, message):
        # Parse the incoming message
        full_data = json.loads(message)

        # Extract relevant data (assuming the Binance 24-hour ticker JSON structure)
        if 's' in full_data:
            formatted_data = {
                # # Trading pair symbol (e.g., BTCUSDT)
                # 'symbol': full_data['s'],
                # # Price change (e.g., -885.51000000)
                # 'priceChange': full_data['p'],
                # # Price change percentage (e.g., -2.004)
                # 'priceChangePercent': full_data['P'],
                # # Weighted average price (e.g., 43715.24834786)
                # 'weightedAvgPrice': full_data['w'],
                # # Previous close price (e.g., 44184.52000000)
                # 'prevClosePrice': full_data['x'],
                # # Last traded price (e.g., 43299.02000000)
                # 'lastPrice': full_data['c'],
                # # Quote volume (e.g., 0.24143000)
                # 'quoteVolume': full_data['Q'],
                # # Current highest bid price (e.g., 43299.02000000)
                # 'bidPrice': full_data['b'],
                # # Current highest bid quantity (e.g., 9.75112000)
                # 'bidQuantity': full_data['B'],
                # # Current lowest ask price (e.g., 43299.03000000)
                # 'askPrice': full_data['a'],
                # # Current lowest ask quantity (e.g., 2.91536000)
                # 'askQuantity': full_data['A'],
                # # Opening price (e.g., 44184.53000000)
                'open': full_data['o'],
                # Highest price in the last 24 hours (e.g., 44279.98000000)
                'high': full_data['h'],
                # Lowest price in the last 24 hours (e.g., 42821.10000000)
                'low': full_data['l'],
                # 24-hour trading volume (e.g., 44685.49662000)
                'volume': full_data['v'],
                # # 24-hour quote asset volume (e.g., 1953437582.29079150)
                # 'quoteAssetVolume': full_data['q'],
                # # Opening time in milliseconds (e.g., 1701867112342)
                # 'openTime': full_data['O'],
                # # Closing time in milliseconds (e.g., 1701953512342)
                # 'closeTime': full_data['C'],
                # # First trade ID (e.g., 3309626308)
                # 'firstTradeId': full_data['F'],
                # # Last trade ID (e.g., 3311120126)
                # 'lastTradeId': full_data['L'],
                # # Number of trades (e.g., 1493819)
                # 'tradeCount': full_data['n']
            }
            self.on_message(formatted_data)
        else:
            print("No market data in message.")

    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")

    def _on_open(self, ws):
        print("WebSocket connection opened")
        # Format the symbol for the Binance stream (e.g., btcusdt@ticker)
        formatted_symbol = f"{self.symbol}@ticker"
        subscribe_message = json.dumps({
            "method": "SUBSCRIBE",
            "params": [formatted_symbol],
            "id": 1  # ID can be any unique integer
        })
        ws.send(subscribe_message)

    def start(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            "wss://stream.binance.com:9443/ws",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        # Run the WebSocket in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.start()

    def stop(self):
        if self.ws:
            self.ws.close()
