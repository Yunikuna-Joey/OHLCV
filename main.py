#* Lumi bot: algo trading framework 
from lumibot.brokers import Alpaca #* broker
from lumibot.backtesting import YahooDataBacktesting    #* framework for backtesting 
from lumibot.strategies.strategy import Strategy        #* bot platform
from lumibot.traders import Trader                      #* Deployment abilities
from datetime import datetime
from timedelta import Timedelta
from alpaca_trade_api import REST 
from util import estimate_sentiment

from dotenv import load_dotenv
load_dotenv()

import os

#* create your .env file and obtain through using os package
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
        # provides the api object into our MLAI strategy 
        self.api = REST(base_url=os.getenv('URL'), key_id=os.getenv('KEY'), secret_key=os.getenv('SECRET'))

    def positionSizing(self):
        # this will allow for the bot to grab its cash value of portfolio
        cash = self.get_cash()
        # this will grab the last known price of the stock ticker 
        lastPrice = self.get_last_price(self.symbol)
        # determining the quantity of shares [rounding down ensuring we don't go over]
        quant = cash * self.cashAtRisk // lastPrice     #* $1000 example => 1000 * 0.5 = 500 / lastPrice of stock == amt of shares 
        return cash, lastPrice, quant

    def getDate(self): 
        # today refers to the day of training / backtesting 
        today = self.get_datetime()
        # back track 4 days 
        priorDays = today - Timedelta(days=4)
        # return both values 
        return today.strftime('%Y-%m-%d'), priorDays.strftime('%Y-%m-%d')

    # def getSentiment(self, tickers): 
    #     data = {}
    #     today, priorDays = self.getDate() 
    #     for ticker in tickers: 
    #         news = self.api.get_news(symbol=ticker, start=priorDays, end=today)
    #         news = [event.__dict__['_raw']['headline'] for event in news]
    #         probability, sentiment = estimate_sentiment(news)
    #         data[ticker] = {'probability': probability, 'sentiment': sentiment}

    #     return data 

    # def on_trading_iteration(self):
    #     array = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
    #     # grab sentiment values for news for these tickers 
    #     values = self.getSentiment(array)

    #     for ticker in array: 
    #         probability, sentiment = values[ticker]['probability'], values[ticker]['sentiment']


    def getSentiment(self): 
        today, priorDays = self.getDate()
        # utilize the alpaca api to 'get news' 
        news = self.api.get_news(symbol=self.symbol, start=priorDays, end=today)                    #* get_news is not supported on crpyto
        # format our news for each news event, obtain the headline from the results above
        news = [event.__dict__['_raw']['headline'] for event in news]
        print(news)
        # return the values of our sentiment and its probability 
        probability, sentiment = estimate_sentiment(news)
        print(probability, sentiment)
        return probability, sentiment

    def helperSentiment(self, ticker): 
        today, priorDays = self.getDate() 
        
        news = self.api.get_news(symbol=ticker, start=priorDays, end=today)
        news = [event.__dict__['_raw']['headline'] for event in news]
        prob, senti = estimate_sentiment(news)

        return prob, senti

    # * every tick of time/ data that is received, a trade can be made
    def on_trading_iteration(self):
        # dynamically cast how much to buy 
        cash, lastPrice, quant = self.positionSizing()
        probability, sentiment = self.getSentiment()

        # helper1 
        prob1, senti1 = self.helperSentiment('GOOGL')
        # helper2 
        prob2, senti2 = self.helperSentiment('MSFT')
        # # helper3
        # prob3, senti3 = self.helperSentiment('NVDA')

        # only purchase if we have enough cash balance
        if cash > lastPrice: 
            # given the sentiment of the news, if it is good and its probability is .999 good, then create a BUY order
            if sentiment == 'positive' and probability > .999: 
                # if there are existing sell orders and the market is positive
                if self.lastTrade == 'sell': 
                    self.sell_all()
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
                    take_profit_price=lastPrice * 1.20, 
                    stop_loss_price=lastPrice * 0.95
                )
                self.submit_order(order)
                # update our action
                self.lastTrade = 'buy'

            elif sentiment == 'negative' and probability > .999:
                if self.lastTrade == 'buy': 
                    self.sell_all()
                order = self.create_order(
                    #i involves the symbol
                    self.symbol, 
                    # amount of shares being sold 
                    quant, 
                    'sell', 
                    type='bracket', 
                    take_profit_price=lastPrice * 0.8, 
                    stop_loss_price=lastPrice * 1.05
                )   
                self.submit_order(order)
                # update our action 
                self.lastTrade = 'sell'

            #* testing new logic that will promote a more proactive approach when encountering neutral sentitment during a time frame
            elif (sentiment == 'neutral' and probability < .750) and ((senti1 == 'positive' and prob1 > .800) or (senti2 == 'positive' and prob2 > .800)): 
                order = self.create_order(
                    self.symbol, 
                    quant, 
                    'buy', 
                    type='bracket', 
                    take_profit_price=lastPrice * 1.25,
                    stop_loss_price=lastPrice * 0.90
                )
                self.submit_order(order)
                self.lastTrade = 'buy'



#* create our broker object 
broker = Alpaca(CRED)
#* create an instance of strategy 
strategy = MLAITrader(name='mlaistrat', broker=broker, parameters={'symbol':'AAPL', 'cashAtRisk': .5})

#* catch time specific time frame to use for testing MLAI
startDate, endDate = datetime(2023, 9, 1), datetime(2023, 12, 31)     #* Y-M-D.. with AAPL, the first news rolled in the year 2015.. test ranges [2015 - 2019] [2021 - 2023]

#* how well our bot is running {comment this line out if you are deploying into live trading}
strategy.backtest(YahooDataBacktesting, startDate, endDate, parameters={'symbol':'AAPL', 'cashAtRisk': .5})

#* ---- TESTING ----
# change symbol in 3 places 
# change cashAtRisk in 3 places 
# change probability on sentiment to act on 
# change stop loss and take profit on BOTH sell and buy orders

#* ------------------- deployment into your brokerage purposes -------------------
#* Create our trader object 
# trader = Trader()
#* Add the strategy into the trader object 
# trader.add_strategy(strategy)
#* deploy 
# trader.run_all()