import datetime
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from functools import total_ordering
from typing import Callable, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finance_data import Finance_Data


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

    def __lt__(self, other: any) -> bool:
        return False if not isinstance(other, _Order) else self.profit < other.profit

    def __add__(self, other: any) -> float:
        if isinstance(other, _Order):
            return self.value() + other.value()
        return self.value() + other

    def __radd__(self, other: any) -> float:
        return self.value() + other

    def fill(self, num_shares, end_t, end_a):
        """fill the order for the shares

        :param num_shares: number of shares
        :param end_t: end time
        :type end_t: datetime
        :param end_a: end amount
        """

        self.num_shares = self.num_shares if num_shares == -1 else num_shares
        self.end_time = end_t
        self.end_amount = end_a
        self.filled = True

    def profit_loss(self):
        """calculates the profit
        """
        try:
            self.profit = (self.end_amount - self.start_amount) * self.num_shares
            return self.profit
        except TypeError:
            logging.error("End or Start amount is None")

    def value(self):
        return self.end_amount if self.filled else self.start_amount


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
        :return number of shares
        :rtype: float
        """

        if num_shares == -1 and self.open_orders:
            order = self.open_orders.popleft()
            order.fill(num_shares, end_t, end_a)
            order.profit_loss()
            self.completed_orders.append(order)
            return order.num_shares

        while num_shares > 0 and self.open_orders:
            order = self.open_orders.popleft()
            if num_shares < order.num_shares:
                replace_order = _Order(
                    order.num_shares - num_shares, order.start_time, order.start_amount
                )
                self.open_orders.appendleft(replace_order)
            order.fill(num_shares, end_t, end_a)
            order.profit_loss()
            self.completed_orders.append(order)

            num_shares -= order.num_shares
        return num_shares

    def order_worth(self) -> float:
        """Returns buying power from orders. Buying power will only work with a starting amount of cash.
        Ex: starting amount (starting amount, not given) - profit from order (completed_order) - invested orders (open_order)

        :return: buying power
        :rtype: float
        """
        return sum(
            (closed_order.profit_loss() for closed_order in self.completed_orders)
        ) - sum(open_order for open_order in self.open_orders)

    def to_df(self) -> pd.DataFrame:
        """
        Converts orders to dataframe
        :return: orders dataframe
        """
        return pd.DataFrame(
            [
                order.__dict__
                for order in self.completed_orders + list(self.open_orders)
            ],
            columns=[
                "num_shares",
                "start_time",
                "start_amount",
                "filled",
                "end_time",
                "end_amount",
                "profit",
            ],
        )


class NoDataException(Exception):
    pass


class Strategey(ABC):
    def __init__(
        self, ticker: str, data: pd.DataFrame = None, initial_amount: int = 100
    ):
        """
        Default Strategy class

        :param ticker: ticker 
        :type ticker: str
        :param data: data for the strategy to find buy and sell points, defaults to None
        :type data: pd.DataFrame, optional
        :param initial_amount: starting amount
        :type inital_amount: int, optional
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
        self.active_orders = 0
        self.orders = Order_Info()
        self.current_amount = initial_amount

    @abstractmethod
    def setup_indicator(self):
        pass

    @abstractmethod
    def buy_and_sell(self):
        pass

    def _curr_amnt(self) -> float:
        """returns buying power at the time when function is called

        :return: buying power
        :rtype: float
        """
        self.current_amount += self.orders.order_worth()
        return self.current_amount

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

    def buy(self, date: datetime, price: float, num_shares: int = -1):
        """Used to buy share at a certain date

        :param date: the day to buy the stock 
        :type date: datetime
        :param price: price of the stock
        :type price: float
        :param num_shares: number of shares to buy
        :type num_shares: int
        """

        current_amount = self._curr_amnt()

        if num_shares == -1 and current_amount > 0:
            num_shares = current_amount // price
        # else:
        #     num_shares = 0

        if current_amount < price * num_shares:
            return
        self.active_orders += num_shares
        self.buy_orders[date] = num_shares
        self.sell_orders[date] = 0
        self.orders.new_order(num_shares, start_t=date, start_a=price)

    def sell(self, date: datetime, price: float, num_shares: int = -1):
        """Used to sell share at a certain date

        :param date: the day to buy the stock 
        :type date: datetime
        :param price: price of the stock
        :type price: float
        :param num_shares: number of shares to buy
        :type num_shares: int
        """

        if self.active_orders > 0:
            if num_shares == -1:
                num_shares = self.orders.close_order(
                    num_shares, end_t=date, end_a=price
                )
            else:
                self.orders.close_order(num_shares, end_t=date, end_a=price)

            self.active_orders -= num_shares
            self.sell_orders[date] = num_shares
            self.buy_orders[date] = 0

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
        plt.legend(["Strategy", "S&P500"])
        if not os.path.isdir("./Graphs"):
            os.makedirs("Graphs")
        plt.savefig("Graphs/" + filename)


class Backtest:
    def __init__(
        self,
        initial_amount: int,
        ticker: str,
        strat: Type[Strategey],
        data_func: Callable = None,
        input_data: pd.DataFrame = None,
        *args,
    ):
        """
        Creates the backtest dataframe with the initial financial data entered
        :param initial_amount: starting amount of money
        :param ticker: ticker
        :param strat: Strategy to backtest
        :param data_func: function to get data if needed
        :param input_data: input data if needed
        :param args: args for the data func
        """

        if not data_func and input_data is None:
            raise NoDataException(
                "There is no default data (one of the OHLCV) provided for the backtest"
            )

        if isinstance(input_data, pd.DataFrame) and not input_data.empty:
            data = input_data
        elif data_func:
            data = data_func(*args)

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

        self.data = data
        self.initial_amount = initial_amount
        self.strat = strat
        self.ticker = ticker

    def setup_strat(self):
        """
        Adds strategy to backtest
        """
        self.strat = self.strat(self.ticker, self.data, self.initial_amount)

    def _enter_positions(self):
        """enters when to buy and sell a stack and calculates total number of stocks owned
        """
        self.backtest["buy"] = pd.Series(self.strat.buy_orders)
        self.backtest["sell"] = pd.Series(self.strat.sell_orders)
        self.backtest.fillna(0, inplace=True)
        self.backtest["shares_owned"] = (
            self.backtest.buy - self.backtest.sell
        ).cumsum()

    def _net_worth(self):
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

        self.setup_strat()
        self._enter_positions()
        self._net_worth()

        market_data = pd.DataFrame(
            {
                "SP500": Finance_Data.market_data.loc[: self.backtest.index[-1]].tail(
                    len(self.backtest)
                )
            }
        )

        self.backtest = pd.concat([self.backtest, market_data], axis=1).fillna(0)
        return self.backtest

    def metrics(self, output: bool = True) -> dict:
        """prints out metrics for the backtest

        :param output: option of whether to print out the stats, defaults to True
        :type output: bool, optional
        :return: stats in the form of a dictionary 
        :rtype: dict
        """

        backtest = self.backtest
        orders = self.strat.orders.to_df()
        start_amount = self.initial_amount
        end_amount = backtest.net_worth[-1]
        time_period = backtest.index

        stats = {}

        stats["Ticker"] = self.ticker.upper()
        stats["Start Time"] = time_period[0]
        stats["End Time"] = time_period[-1]
        stats["Start Amount"] = start_amount
        stats["End Amount"] = end_amount

        # ------ Average Hold Time -----
        stats["Average Hold Time"] = str((orders.end_time - orders.start_time).mean())

        # ------ Average Losses -----
        stats["Average Losses"] = orders.loc[orders.profit < 0].profit.mean()

        # ------ Average Profits -----
        stats["Average Profits"] = orders.loc[orders.profit > 0].profit.mean()

        # ------ Biggest Loss -----
        stats["Biggest Loss"] = orders.loc[orders.profit < 0].profit.min()

        # ------ Biggest Win -----
        stats["Biggest Win"] = orders.profit.max()

        # ------ CAGR -----
        years = (time_period[-1] - time_period[0]).days // 365
        cagr = ((end_amount / start_amount) ** (1 / years) - 1) * 100

        stats["Compound Annual Growth Rate (%) "] = cagr

        # ------------  Max Drawdown -----------
        rolling_max = backtest.net_worth.cummax()
        drawdown = backtest.net_worth / rolling_max - 1

        stats["Max Drawdown (%)"] = drawdown.min() * 100
        stats["Average Drawdown (%)"] = drawdown.mean() * 100

        # ------------  Net Profit -----------
        stats["Net Profit"] = backtest.net_worth[-1] - start_amount

        # ------------  Profit Factor -----------
        loss = orders.loc[orders.profit < 0].profit.sum()
        profit = orders.loc[orders.profit > 0].profit.sum()
        if loss == 0 or np.isnan(loss):
            loss = -1
        stats["Profit Factor"] = profit / -loss

        # ------------  Risk Reward -----------
        if not orders.empty:
            total_gain = orders.groupby("filled").profit.sum()[1]
            total_risked = (orders.start_amount * orders.num_shares).sum()
            risk_reward = total_gain / total_risked
        else:
            risk_reward = np.NaN

        stats["Risk Reward"] = risk_reward

        # ------------  Sharpe Ratio -----------
        risk_free_rate = Finance_Data.risk_free_rate_10y
        annual_er = (backtest.net_worth.pct_change().mean() + 1) ** 255 - 1
        sharpe = (annual_er - risk_free_rate) / (
            backtest.net_worth.pct_change().std() * np.sqrt(252)
        )
        stats["Sharpe Ratio"] = sharpe

        # ------------  Volatility -----------
        stats[
            "Volatility Annualized (% change)"
        ] = backtest.net_worth.pct_change().std() * np.sqrt(252)

        # ------------  Beta -----------

        strat_r = (backtest.net_worth.pct_change()).mean()
        market_r = (backtest.SP500.pct_change()).mean()
        covariance = (
            (backtest.net_worth.pct_change() - strat_r)
            * (backtest.SP500.pct_change() - market_r)
        ).sum() / len(backtest)

        variance = backtest.net_worth.pct_change().var()

        stats["Beta"] = covariance / variance

        # ------------  Alpha -----------
        stock_return = (
            backtest.net_worth[-1] - backtest.net_worth[0]
        ) / backtest.net_worth[0]

        alpha = (
            stock_return
            - Finance_Data.risk_free_rate_10y
            - stats["Beta"]
            * (
                (backtest.SP500[-1] / backtest.SP500[0] - 1)
                - Finance_Data.risk_free_rate_10y
            )
        )

        stats["Alpha"] = alpha

        # ------------  R-Squared -----------
        stats["R-Squared"] = covariance / (
            np.sqrt(variance) * backtest.SP500.pct_change().std()
        )

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


if __name__ == "__main__":
    pass
