import datetime
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from functools import total_ordering
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finance_data import Finance_Data, download_data, load_data


@total_ordering
class _Order:
    def __init__(self, num_shares, start_t=None, start_a=None):
        """order class for purchase of shares

        :param num_shares: number of shares to buy 
        :type num_shares: int
        :param start_t: start time, defaults to None
        :type start_t: datetime, optional
        :param start_a: start amount, defaults to None
        :type start_a: float, optional
        """
        self.profit = None
        self.end_amount = None
        self.end_time = None
        self.num_shares = num_shares
        self.start_time = start_t
        self.start_amount = start_a
        self.filled = False

    def __lt__(self, other):
        return False if not isinstance(other, _Order) else self.profit < other.profit

    def fill(self, num_shares, end_t, end_a):
        """fill the order for the shares

        :param num_shares: number of shares
        :param end_t: end time
        :type end_t: datetime
        :param end_a: end amount
        """
        self.num_shares = num_shares
        self.end_time = end_t
        self.end_amount = end_a
        self.filled = True

    def profit_loss(self):
        """calculates the profit
        """
        try:
            self.profit = self.end_amount - self.start_amount
        except TypeError:
            logging.error("End or Start amount is None")


class Order_Info:
    def __init__(self):
        self.open_orders = deque()
        self.completed_orders = []
        self.total_orders = 0

    def new_order(self, num_shares, start_t, start_a):
        """
        Creates new Order
        :param num_shares: number of shares
        :param start_t: start time
        :param start_a: start amount
        """
        order = _Order(num_shares, start_t, start_a)
        self.open_orders.append(order)
        self.total_orders += 1

    def close_order(self, num_shares, end_t, end_a):
        """
        Closes order and moves order to completed list. Fills end time and end amount
        :param num_shares: number of shares to fill
        :param end_t: end time
        :param end_a: end amount
        """
        while num_shares > 0:
            order = self.open_orders.popleft()
            if num_shares < order.num_shares:
                # Num shares less than order shares
                replace_order = _Order(
                    order.num_shares - num_shares, order.start_time, order.start_amount
                )
                self.open_orders.appendleft(replace_order)
            order.fill(num_shares, end_t, end_a)
            order.profit_loss()
            self.completed_orders.append(order)

            num_shares -= order.num_shares

    def to_df(self) -> pd.DataFrame:
        """
        Converts orders to dataframe
        :return: orders dataframe
        """
        return pd.DataFrame(
            [order.__dict__ for order in self.completed_orders + list(self.open_orders)]
        )


class NoDataException(Exception):
    pass


class Strategey(ABC):
    def __init__(self, ticker: str, data: pd.DataFrame = None):
        """
        Default Strategy class

        :param ticker: ticker 
        :type ticker: str
        :param data: data for the strategy to find buy and sell points, defaults to None
        :type data: pd.DataFrame, optional
        """

        try:
            if isinstance(data, pd.DataFrame) and not data.empty:
                self.data = data
            else:
                self.data = Finance_Data(ticker).data
        except NameError:
            logging.error("There must be one type of (OHLCV) data for the strategy")

        self.indicators = []
        self.ticker = ticker.upper()
        self.buy_orders = {}
        self.sell_orders = {}
        self.orders = Order_Info()

    @abstractmethod
    def setup_indicator(self):
        pass

    @abstractmethod
    def buy_and_sell(self):
        pass

    def rolling_average(self, data, time_frame):
        """Generates simple moving average for data

        :param data: data that the sma should be generated for (needs date index) 
        :type data: DataFrame, Series 
        :param time_frame: time frame for sma (e.g "20D") 
        :type time_frame: str 
        :return: sma for data for a specific time frame 
        :rtype: Series 
        """
        return data.rolling(time_frame).mean()

    def buy(self, num_shares: int, date: datetime, price: float):
        """Used to buy share at a certain date

        :param num_shares: number of shares to buy
        :type num_shares: int
        :param date: the day to buy the stock 
        :type date: datetime
        :param price: price of the stock
        :type price: float
        """

        self.buy_orders[date] = num_shares
        self.sell_orders[date] = 0
        self.orders.new_order(num_shares, start_t=date, start_a=price)

    def sell(self, num_shares: int, date: datetime, price: float):
        """Used to sell share at a certain date

        :param num_shares: number of shares to buy
        :type num_shares: int
        :param date: the day to buy the stock 
        :type date: datetime
        :param price: price of the stock
        :type price: float
        """

        self.sell_orders[date] = num_shares
        self.buy_orders[date] = 0
        self.orders.close_order(num_shares, end_t=date, end_a=price)

    def plot_data(
        self,
        data,
        title="Stocks",
        xlabel="Date",
        ylabel="Return",
        filename="data.png",
        color="LIGHT",
        area=False,
    ):
        """Plots data nicely

        :param data: data to be plotted 
        :type data: DataFrame or Series with date index 
        :param title: title of plot, defaults to "Stocks"
        :type title: str, optional
        :param xlabel: x-axis label, defaults to "Date"
        :type xlabel: str, optional
        :param ylabel: y-axis label, defaults to "Return"
        :type ylabel: str, optional
        :param filename: output filename (will automatically be stored in a folder called Graphs), defaults to "data.png"
        :type filename: str, optional
        :param color: LIGHT or DARK color graph, defaults to "LIGHT"
        :type color: str, optional
        :param area: enables area type graph, defaults to False
        :type area: bool, optional
        """
        light_style = "graph_colors/stock-light.mplstyle"
        dark_style = "graph_colors/stock-dark.mplstyle"
        text_color = "black"
        if color == "DARK":
            plt.style.use(dark_style)
            text_color = "white"
        else:
            plt.style.use(light_style)
        ax = data.plot.area(stacked=False, zorder=10) if area else data.plot(zorder=10)
        ax.grid(zorder=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.text(
            0.5,
            0.5,
            self.ticker,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=76,
            weight="bold",
            alpha=0.3,
            color=text_color,
            variant="small-caps",
            zorder=3,
        )
        if not os.path.isdir("./Graphs"):
            os.makedirs("Graphs")
        plt.savefig("Graphs/" + filename)


class Backtest:
    def __init__(
        self,
        initial_amount: int,
        ticker: str,
        data_func: Callable = None,
        input_data: pd.DataFrame = None,
        *args,
        **kwargs
    ):
        """
        Creates the backtest dataframe with the initial financial data entered
        :param initial_amount: starting amount of money
        :param ticker: ticker
        :param data_func: function to get data if needed
        :param input_data: input data if needed
        :param args: args for the data func
        :param kwargs: for individual column data
        """

        self.initial_amount = initial_amount
        self.strat = None

        if not bool(kwargs) and not data_func and input_data is None:
            raise NoDataException(
                "There is no default data (one of the OHLCV) provided for the backtest"
            )

        if isinstance(input_data, pd.DataFrame) and not input_data.empty:
            data = input_data
        elif data_func:
            data = data_func(*args)[ticker]
        else:
            data = kwargs

        if isinstance(data, pd.DataFrame):
            data.columns = data.columns.str.lower()

        self.backtest = pd.DataFrame(
            data,
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "net_worth",
                "shares_owned",
                "buy",
                "sell",
            ],
        )

    def setup_strat(self, strat: Strategey):
        """
        Adds strategy to backtest
        :param strat: Strategy
        :type strat: Strategey
        """
        self.strat = strat

    def _calc_trading_shares(self):
        """enters when to buy and sell a stack and calculates total number of stocks owned
        """
        self.backtest["buy"] = pd.Series(self.strat.buy_orders)
        self.backtest["sell"] = pd.Series(self.strat.sell_orders)
        self.backtest.fillna(0, inplace=True)
        self.backtest["shares_owned"] = (
            self.backtest.buy - self.backtest.sell
        ).cumsum()

    def _calculate_net_worth(self):
        """Calculates net worth from the strategy backtested
        """
        buy = self.backtest.buy
        sell = self.backtest.sell
        shares_owned = self.backtest.shares_owned

        current_net_worth = self.initial_amount
        real_price = self.backtest.close

        net_worth = []
        buy_cost = 0

        for i in range(len(self.backtest)):

            buy_cost += buy[i] * real_price[i]
            if sell[i] != 0:
                current_net_worth += sell[i] * real_price[i]
            net_worth.append(
                current_net_worth + real_price[i] * shares_owned[i] - buy_cost
            )

        self.backtest["net_worth"] = net_worth

    def run(self) -> pd.DataFrame:
        """Runs the backtest and fills out backtest DataFrame 

        :return: backtest data 
        :rtype: DataFrame 
        """

        self._calc_trading_shares()
        self._calculate_net_worth()

        self.backtest = pd.concat(
            [
                self.backtest,
                pd.DataFrame(
                    {
                        "SP500": Finance_Data.market_data.loc[
                            : self.backtest.index[-1]
                        ].tail(len(self.backtest))
                    }
                ),
            ],
            axis=1,
        ).fillna(0)
        return self.backtest

    def metrics(self, output: bool = True) -> dict:
        """prints out metrics for the backtest

        :param output: option of whether to print out the stats, defaults to True
        :type output: bool, optional
        :return: stats in the form of a dictionary 
        :rtype: dict
        """

        orders = self.strat.orders.to_df()

        loss = orders.loc[orders.profit < 0].profit.sum()
        if loss == 0 or np.isnan(loss):
            loss = -1

        # ! WILL ADD MORE STATS LATER
        stats = {
            "Net Profit": self.backtest.net_worth[-1],
            "Profit Factor": orders.loc[orders.profit > 0].profit.sum() / -loss,
            "Biggest Win": orders.profit.max(),
            "Biggest Loss": orders.loc[orders.profit < 0].profit.min(),
            "Risk Reward": orders.groupby("filled").profit.sum()[1]
            / orders.groupby("filled").start_amount.sum()[1],
            "Average Profits": orders.profit.mean(),
            "Average Losses": orders.loc[orders.profit < 0].profit.mean(),
            "Average Hold Time": str((orders.end_time - orders.start_time).mean()),
        }

        if output:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                150,
                "display.precision",
                3,
            ):
                print(pd.DataFrame(stats, index=["Stats"]).T)

        return stats


class MA_Cross_Strat(Strategey):
    def __init__(self, ticker: str, data: pd.DataFrame):
        """Strategy that buys stock when a short moving average crosses a long one and sells when the long crosses the short.
        20 day crossing 100 and vise versa. Child class of Strategy

        :param ticker: ticker symbol 
        :type ticker: str 
        :param data: data that the strategy will be tested on
        :type data: pd.DataFrame
        """
        Strategey.__init__(self, ticker, data)
        self.setup_indicator()
        self.buy_and_sell()

    def setup_indicator(self):
        """Configures and sets up the indicators. Adds indicators that strategey needs to the DataFrame for the ticker
        """
        self.indicators.append(self.rolling_average(self.data, 20).Close)
        self.indicators.append(self.rolling_average(self.data, 100).Close)

    def buy_and_sell(self):
        """Main strategey is in this function. Decides when to mark a day as buy and sell based on the indicators
        """
        twenty_ma, hundred_ma = self.indicators

        cross = twenty_ma > hundred_ma

        buy = np.where(((cross != cross.shift(1)) & cross))[0]
        sell = np.where((cross != cross.shift(1)) & (cross == False))[0]

        first_buy = buy[0]

        for i in buy:
            data = self.data.iloc[i]
            self.buy(1, data.name, data.Close)

        for i in sell:
            data = self.data.iloc[i]
            if i > first_buy:
                self.sell(1, data.name, data.Close)


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
        self.indicators.append(self.data.Close * 1.05)
        self.indicators.append(self.data.Close * 0.99)

    def buy_and_sell(self):
        sell_price, buy_price = self.indicators
        current_amount_idx = 0
        last_move_sell = False

        self.buy(1, self.data.index[0], self.data.Close[0])

        for i in range(1, len(self.data)):
            date = self.data.index[i]
            value = self.data.Close[i]

            if (value >= sell_price[current_amount_idx]) and not last_move_sell:
                self.sell(1, date, value)
                current_amount_idx = i
                last_move_sell = True
            elif (value <= buy_price[current_amount_idx]) and last_move_sell:
                self.buy(2, date, value)
                current_amount_idx = i
                last_move_sell = False


if __name__ == "__main__":

    # Download data for tickers
    # download_data('AAPL', 'MSFT', 'TSLA').AAPL.to_csv("./data/aapl.csv")

    # Load data from a directory
    data = load_data("./data")

    # Enter data into strategy
    tps = MA_Cross_Strat("aapl", data["aapl"])

    # Initalize backtest
    t = Backtest(100, "aapl", input_data=data["aapl"])
    t.setup_strat(tps)

    output = t.run()
    t.metrics()
    # orders = t.strat.orders
    # orders.to_df().to_csv("orders.csv")

    tps.plot_data(
        ((output[["net_worth", "SP500"]].last("10Y").pct_change() + 1).cumprod() * 100)
        - 100,
        title="Percent return of MA strategy against time",
        ylabel="Percent Returns",
        color="LIGHT",
    )
