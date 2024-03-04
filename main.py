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
    def initialize(self, symbol:str='AAPL', cashAtRisk:float=.5): 
        self.symbol = symbol
        # how frequent are we trading 
        self.sleeptime = '24H'
        self.lastTrade = None
        # the ammount of cash our bot is willing to risk, .5 == half of our portfolio cash balance 
        self.cashAtRisk = cashAtRisk

    def positionSizing(self):
        # this will allow for the bot to grab its cash value of portfolio
        cash = self.get_cash()
        # this will grab the last known price of the stock ticker 
        lastPrice = self.get_last_price(self.symbol)
        # determining the quantity of shares [rounding down ensuring we don't go over]
        quant = cash * self.cashAtRisk // lastPrice     #* $1000 example => 1000 * 0.5 = 500 / lastPrice of stock == amt of shares 
        return cash, lastPrice, quant

    #* every tick of time/ data that is received, a trade can be made
    def on_trading_iteration(self):
        # dynamically cast how much to buy 
        cash, lastPrice, quant = self.positionSizing()

        # only purchase if we have enough cash balance
        if cash > lastPrice: 
            if self.lastTrade == None: 
                # creating the order
                order = self.create_order(
                    # involves the symbol
                    self.symbol, 
                    # amount of shares being purchased 
                    quant, 
                    # either buy or sell order
                    'buy', 
                    # market, 
                    # limit, 
                    #* bracket has a lower and upper bound for stop loss and take profit respectively   
                    type='bracket',            
                    take_profit_price=lastPrice*1.20, 
                    stop_loss_price=lastPrice*0.95
                )
                self.submit_order(order)
                self.lastTrade = 'buy'

#* create our broker object 
broker = Alpaca(CRED)
#* create an instance of strategy 
strategy = MLAITrader(name='mlaistrat', broker=broker, parameters={'symbol':'AAPL', 'cashAtRisk': .5})

#* catch time specific time frame to use for testing MLAI
startDate, endDate = datetime(2023, 1, 1), datetime(2023, 12, 31)     #* Y-M-D
#* how well our bot is running {guess}
strategy.backtest(YahooDataBacktesting, startDate, endDate, parameters={'symbol':'AAPL', 'cashAtRisk': .5})