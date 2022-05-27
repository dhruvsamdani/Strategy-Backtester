import pandas as pd

from strats import Strategey


class MA_Cross_Strat(Strategey):
    def __init__(self, ticker: str, data: pd.DataFrame, initial_amount=100):
        """Strategy that buys stock when a short moving average crosses a long one and sells when the long crosses the short.
        20 day crossing 100 and vise versa. Child class of Strategy

        :param ticker: ticker symbol 
        :type ticker: str 
        :param data: data that the strategy will be tested on
        :type data: pd.DataFrame
        """
        Strategey.__init__(self, ticker, data, initial_amount=initial_amount)
        self.setup_indicator()
        self.buy_and_sell()

    def setup_indicator(self):
        """Configures and sets up the indicators. Adds indicators that strategey needs to the DataFrame for the ticker
        """
        self.indicators.append(self.rolling_average(self.data, 20).close)
        self.indicators.append(self.rolling_average(self.data, 100).close)

    def buy_and_sell(self):
        """Main strategey is in this function. Decides when to mark a day as buy and sell based on the indicators
        """
        twenty_ma, hundred_ma = self.indicators

        cross = twenty_ma > hundred_ma

        buy = cross.loc[((cross != cross.shift(1)) & cross)].rename("buy")
        sell = cross.loc[(cross != cross.shift(1)) & (cross == False)].rename("sell")

        first_buy = buy.index[0]

        trade = pd.concat([buy, sell], axis=1)

        for i in trade.index:
            data = self.data.loc[i]
            if trade.buy.loc[i] == True:
                self.buy(data.name, data.close)
            else:
                if i > first_buy:
                    self.sell(data.name, data.close)


# ! NOT WORKING RIGHT NOW
class Ten_Percent_Strat(Strategey):
    """Example expiramental strategey
    Buys when price goes 1% below last sell price and sells when price
    goes above 5% of last buy price

    """

    def __init__(self, ticker, data):
        Strategey.__init__(self, ticker, data)
        self.setup_indicator()
        self.buy_and_sell()

    def setup_indicator(self):
        self.indicators.append(self.data.close * 1.05)
        self.indicators.append(self.data.close * 0.99)

    def buy_and_sell(self):
        sell_price, buy_price = self.indicators
        current_amount_idx = 0
        last_move_sell = False

        self.buy(1, self.data.index[0], self.data.close[0])

        for i in range(1, len(self.data)):
            date = self.data.index[i]
            value = self.data.close[i]

            if (value >= sell_price[current_amount_idx]) and not last_move_sell:
                self.sell(1, date, value)
                current_amount_idx = i
                last_move_sell = True
            elif (value <= buy_price[current_amount_idx]) and last_move_sell:
                self.buy(2, date, value)
                current_amount_idx = i
                last_move_sell = False

