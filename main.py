#* Lumi bot: algo trading framework 
from lumibot.brokers import Alpaca #* broker
from lumibot.backtesting import YahooDataBacktesting    #* framework for backtesting 
from lumibot.strategies.strategy import Strategy        #* bot platform
from lumibot.traders import Trader                      #* Deployment abilities
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import os

CRED = {
    'API_KEY': os.getenv('KEY'), 
    'API_SECRET': os.getenv('SECRET'), 
    'PAPER': True
}

#* trading logic here 
class MLAITrader(Strategy):
    #* setup 
    def initialize(self, symbol:str='AAPL'): 
        self.symbol = symbol
        # how frequent are we trading 
        self.sleeptime = '24H'
        self.lastTrade = None


    #* every tick of time/ data that is received, a trade can be made
    def on_trading_iteration(self):
        if self.lastTrade == None: 
            # creating the order
            order = self.create_order(
                # involves the symbol
                self.symbol, 
                # amount of shares being purchased 
                20, 
                # either buy or sell order
                'buy', 
                # market, limit, bracket... 
                type='market'
            )
            self.submit_order(order)
            self.lastTrade = 'buy'

#* create our broker object 
broker = Alpaca(CRED)
#* create an instance of strategy 
strategy = MLAITrader(name='mlaistrat', broker=broker, parameters={'symbol':'AAPL'})

#* catch time specific time frame to use for testing MLAI
startDate, endDate = datetime(2023, 1, 1), datetime(2023, 12, 31)     #* Y-M-D
#* how well our bot is running {guess}
strategy.backtest(YahooDataBacktesting, startDate, endDate, parameters={'symbol':'AAPL'})