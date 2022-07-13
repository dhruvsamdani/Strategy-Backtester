import numpy as np
import pandas as pd

from strat_backtest.backtest import Strategy


class MA_Cross_Strat(Strategy):
    def __init__(
        self,
        ticker: str,
        data: pd.DataFrame,
        initial_amount: int = 100,
        fast: int = 20,
        lagging: int = 100,
    ):
        """Strategy that buys stock when a short moving average crosses a long one and sells when the long crosses the short.
        20 day crossing 100 and vise versa. Child class of Strategy

        :param ticker: ticker symbol
        :type ticker: str
        :param data: data that the strategy will be tested on
        :type data: pd.DataFrame
        :param initial_amount: initial starting amount
        :type initial_amount: int
        :param fast: faster moving average
        :type fast: int
        :param lagging: slower moving average
        :type lagging: int
        """
        Strategy.__init__(self, ticker, data, initial_amount=initial_amount)
        self.fast = fast
        self.lagging = lagging
        self.setup_indicator()
        self.buy_and_sell()

    def setup_indicator(self):
        """Configures and sets up the indicators. Adds indicators that strategey needs to the DataFrame for the ticker"""
        self.indicators.append(self.data.close.rolling(self.fast).mean())
        self.indicators.append(self.data.close.rolling(self.lagging).mean())

    def buy_and_sell(self):
        """Main strategey is in this function. Decides when to mark a day as buy and sell based on the indicators"""
        twenty_ma, hundred_ma = self.indicators

        cross = twenty_ma > hundred_ma

        buy = cross.iloc[np.where(cross & (cross != cross.shift(1)))].rename("buy")
        sell = cross.iloc[np.where(~cross & (cross != cross.shift(1)))].rename("sell")

        if buy.empty:
            return
        first_buy = buy.index[0]

        trade = pd.concat([buy, sell], axis=1)

        for i, close in zip(trade.index, self.data.close[trade.index]):
            if trade.buy.loc[i] == True:
                self.buy(i, close)

                # self.buy(i, close, stop_loss=close * 0.90)
            elif i > first_buy:
                self.sell(i, close)


class Ten_Percent_Strat(Strategy):
    """Example expiramental strategey
    Buys when price goes 1% below last sell price and sells when price
    goes above 5% of last buy price"""

    def __init__(
        self, ticker, data, initial_amount: int = 100, sell: int = 1.05, buy: int = 0.99
    ):
        Strategy.__init__(self, ticker, data, initial_amount=initial_amount)
        self.sell_price = sell
        self.buy_price = buy
        self.setup_indicator()
        self.buy_and_sell()

    def setup_indicator(self):
        self.indicators.append(self.data.close * self.sell_price)
        self.indicators.append(self.data.close * self.buy_price)

    def buy_and_sell(self):
        sell_price, buy_price = self.indicators
        current_amount_idx = 0
        last_move_sell = False

        self.buy(self.data.index[0], self.data.close[0])

        for i in range(1, len(self.data)):
            date = self.data.index[i]
            value = self.data.close[i]

            if (value >= sell_price[current_amount_idx]) and not last_move_sell:
                self.sell(date, value)
                current_amount_idx = i
                last_move_sell = True
            elif (value <= buy_price[current_amount_idx]) and last_move_sell:
                self.buy(date, value)
                current_amount_idx = i
                last_move_sell = False
