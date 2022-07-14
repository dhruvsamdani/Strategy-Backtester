import datetime
import heapq
import logging
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import total_ordering
from numbers import Number
from typing import Any, Callable, List, Optional, Tuple, Type
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strat_backtest.backtest import Optimize
from strat_backtest.backtest.finance_data import Finance_Data


@total_ordering
class _Order:
    def __init__(
        self,
        num_shares: float,
        id: uuid.UUID,
        stop_loss: Optional[float] = None,
        start_t: datetime.datetime = None,
        start_a: float = None,
    ):
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
        self.uuid = id
        self.stop_loss = stop_loss

    def __lt__(self, other: Any) -> bool:
        return False if not isinstance(other, _Order) else self.profit < other.profit

    def __add__(self, other: Any) -> float:
        if isinstance(other, _Order):
            return self.value() + other.value()
        return self.value() + other

    def __radd__(self, other: Any) -> float:
        return self.value() + other

    def __mul__(self, other: Any) -> float:
        if isinstance(other, _Order):
            return self.num_shares * other.num_shares
        return self.num_shares * other

    __rmul__ = __mul__

    def fill(self, num_shares: float, end_t: datetime, end_a: float):
        """fill the order for the shares

        :param num_shares: number of shares
        :type num_shares: float
        :param end_t: end time
        :type end_t: datetime
        :param end_a: end amount
        :type end_a: float
        """

        self.num_shares = self.num_shares if num_shares == -1 else num_shares
        self.end_time = end_t
        self.end_amount = end_a
        self.filled = True

    def profit_loss(self):
        """calculates the profit"""
        try:
            profit = (self.end_amount - self.start_amount) * self.num_shares
            self.profit = profit
            return profit
        except TypeError:
            logging.error("End or Start amount is None")

    def value(self):
        """how much the order is worth (buy or sell ammount)"""
        return self.end_amount if self.filled else self.start_amount


@dataclass(order=True)
class _Order_Packet:
    """Dataclass for multiple orders being executed on one day"""

    orders: list = field(default_factory=list[_Order])
    sort_index: int = field(init=False)
    num_shares: int = 0

    def __mul__(self, other):
        if isinstance(other, Number):
            return self.num_shares * other
        raise ArithmeticError("_Order_Packet can only be multiplied with a number")

    def __add__(self, other):
        if isinstance(other, Number):
            return self.num_shares + other

    def __sub__(self, other):
        if isinstance(other, Number):
            return self.num_shares - other

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__

    def __post_init__(self):
        self.sort_index = self.num_shares

    def append(self, item: _Order):
        self.num_shares += item.num_shares
        self.orders.append(item)


class Order_Manager:
    def __init__(self):
        self.open_orders = deque()
        self.completed_orders = []
        self.shares_owned = {}
        self.total_orders = 0
        self.total_shares = 0
        self.orders = defaultdict(uuid.UUID)

    def _replace_order(self, order: _Order, num_shares: float) -> _Order:
        id = uuid.uuid1()
        replace_order = _Order(
            order.num_shares - num_shares,
            id,
            start_t=order.start_time,
            start_a=order.start_amount,
            stop_loss=order.stop_loss,
        )
        self.open_orders.appendleft(replace_order)
        self.orders[id] = replace_order
        return replace_order

    def new_order(
        self,
        num_shares: float,
        id: uuid.UUID,
        start_t: datetime.datetime,
        start_a: float,
        stop_loss: Optional[float] = None,
    ) -> _Order:
        """
        Creates new Order
        :param num_shares: number of shares
        :param start_t: start time
        :param start_a: start amount
        """
        order = _Order(
            num_shares, id, stop_loss=stop_loss, start_t=start_t, start_a=start_a
        )
        self.open_orders.append(order)
        self.total_orders += 1
        self.total_shares += num_shares
        self.shares_owned[start_t] = self.total_shares
        self.orders[id] = order
        return order

    def close_order(
        self, num_shares: float, end_t: datetime.datetime, end_a: float
    ) -> Tuple[float, List[uuid.UUID]]:
        """
        Closes order and moves order to completed list. Fills end time and end amount
        :param num_shares: number of shares to fill
        :param end_t: end time
        :param end_a: end amount
        :return number of shares
        :rtype: float
        """

        completed = _Order_Packet()
        if num_shares == -1 and self.open_orders:
            order = self.open_orders.popleft()
            order.fill(num_shares, end_t, end_a)
            order.profit_loss()
            self.completed_orders.append(order)
            self.total_shares -= order.num_shares
            self.shares_owned[end_t] = self.total_shares
            return order

        while num_shares > 0 and self.open_orders:
            order = self.open_orders.popleft()
            if num_shares < order.num_shares:
                replace_order = self._replace_order(order, num_shares)
                self.open_orders.appendleft(replace_order)
            order.fill(num_shares, end_t, end_a)
            order.profit_loss()
            completed.append(order)
            self.completed_orders.append(order)
            self.total_shares -= num_shares
            num_shares -= order.num_shares
        self.shares_owned[end_t] = self.total_shares
        return completed

    def order_worth(self) -> float:
        """Returns buying power from orders. Buying power will only work with a starting amount of cash.
        Ex: starting amount (starting amount, not given) - profit from order (completed_order) - invested orders (open_order)

        :return: buying power
        :rtype: float
        """
        return sum(
            closed_order.profit_loss() for closed_order in self.completed_orders
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


class Strategy(ABC):
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

        self._num_sell = {}
        self.indicators = []
        self.ticker = ticker.upper()
        self.buy_orders = {}
        self.sell_orders = {}
        self.active_orders = 0
        self.orders = Order_Manager()
        self.current_amount = initial_amount
        self.stop_loss = []

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

    def _exit_stop_loss(self, trading_date: pd.Timestamp) -> tuple:
        """If there is a stop loss that needs to be executed this function will tell the user where

        :param trading_date: current trading date
        :return: None tuple or tuple of stop loss, stop loss date, and the uuid for the order
        """
        if (
            not self.stop_loss
            or self.orders.orders[self.stop_loss[0][1]].start_time > trading_date
        ):
            return (None,)

        sl, uid = heapq.heappop(self.stop_loss)

        date = self.orders.orders[uid].start_time

        stop_loss_exit = self.data.close.loc[
            (self.data.close <= sl)
            & (self.data.index >= date)
            & (self.data.index < trading_date)
        ]

        if stop_loss_exit.empty:
            return (None,)
        return stop_loss_exit.iloc[0], stop_loss_exit.index[0], uid

    def _sell_functionality(
        self, shares: float, end_time: pd.Timestamp, end_amount: float
    ):
        """Functionality for a sell order

        :param shares: number of shares
        :param end_time: end time of sell order
        :param end_amount: end amount of sell order
        """
        closed_orders = self.orders.close_order(
            shares, end_t=end_time, end_a=end_amount
        )
        self.active_orders -= closed_orders.num_shares
        self.sell_orders[end_time] = closed_orders

    def buy(
        self,
        date: datetime.datetime,
        price: float,
        num_shares: float = -1,
        stop_loss: Optional[float] = None,
    ):
        """Used to buy share at a certain date

        :param date: the day to buy the stock
        :type date: datetime
        :param price: price of the stock
        :type price: float
        :param num_shares: number of shares to buy
        :type num_shares: float

        if num_shares is -1 then the max amount of stocks will be bought

        Stop Loss
        ---------
        stop loss must be a number not a percentage
        Stop Loss Usage ::

            # 90 percent stop loss
            self.buy(date, price, num_shares, stop_loss=0.9 * price)

        """

        sl = self._exit_stop_loss(date)
        while sl[0] is not None:
            self._sell_functionality(
                min(self.orders.orders[sl[2]].num_shares, -1), sl[1], sl[0]
            )
            sl = self._exit_stop_loss(date)

        uid = uuid.uuid1()
        current_amount = self._curr_amnt()

        if num_shares == -1 and current_amount > 0:
            num_shares = current_amount // price

        if current_amount < price * num_shares:
            return

        order = self.orders.new_order(
            num_shares, uid, stop_loss=stop_loss, start_t=date, start_a=price
        )
        self.active_orders += num_shares
        self.buy_orders[date] = order

        if stop_loss is not None:
            sl = (order.stop_loss, order.uuid)
            heapq.heappush(self.stop_loss, sl)

    def sell(self, date: datetime, price: float, num_shares: float = -1):
        """Used to sell share at a certain date

        :param date: the day to buy the stock
        :type date: datetime
        :param price: price of the stock
        :type price: float
        :param num_shares: number of shares to buy
        :type num_shares: float

        if num_shares is -1 then the max amount of stocks will be sold
        """

        stop_loss = self._exit_stop_loss(date)
        while stop_loss[0] is not None:
            self._sell_functionality(
                min(self.orders.orders[stop_loss[2]].num_shares, -1),
                stop_loss[1],
                stop_loss[0],
            )
            stop_loss = self._exit_stop_loss(date)

        if self.active_orders > 0:
            self._sell_functionality(num_shares, date, price)

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

        with resources.path(
            "strat_backtest.graph_colors", "stock-light.mplstyle"
        ) as light:
            light_style = light
        with resources.path(
            "strat_backtest.graph_colors", "stock-dark.mplstyle"
        ) as dark:
            dark_style = dark
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
        strat: Type[Strategy],
        *args,
        data_func: Callable = None,
        input_data: pd.DataFrame = None,
        **kwargs,
    ):
        """
        Creates the backtest dataframe with the initial financial data entered
        :param initial_amount: starting amount of money
        :param ticker: ticker
        :param strat: Strategy to backtest
        :param data_func: function to get data if needed
        :param input_data: input data if needed
        :param args: args for the data func
        :param kwargs: kwargs for the strategy

        Usage
        -----
        ::

        Backtest(5000, "AAPL", MA_Cross_Strat, input_data=data, fast=20, lagging=100)
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
        self.kwargs = kwargs

    def setup_strat(self):
        """Adds strategy to backtest"""
        self.strat = self.strat(
            self.ticker, self.data, self.initial_amount, **self.kwargs
        )

    def _enter_positions(self):
        """enters when to buy and sell a stack and calculates total number of stocks owned"""
        self.backtest["buy"] = pd.Series(self.strat.buy_orders, dtype=object)
        self.backtest["sell"] = pd.Series(self.strat.sell_orders, dtype=object)
        self.backtest[["buy", "sell"]] = self.backtest[["buy", "sell"]].fillna(0)
        self.backtest["shares_owned"] = pd.Series(
            self.strat.orders.shares_owned, dtype=float
        ).fillna(method="ffill")
        self.backtest["shares_owned"] = self.backtest.shares_owned.ffill().fillna(0)

    def _net_worth(self):
        """Calculates net worth from the strategy backtested"""
        close = self.backtest.close
        cost_adjusted_buy = (self.backtest.buy * close).cumsum()
        cost_adjusted_sell = (self.backtest.sell * close).cumsum()
        cost_adjusted_shares_owned = self.backtest.shares_owned * close

        self.backtest["net_worth"] = (
            cost_adjusted_shares_owned
            - cost_adjusted_buy
            + cost_adjusted_sell
            + self.initial_amount
        )

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

        self.backtest = pd.concat([self.backtest, market_data], axis=1)
        return self.backtest

    def optimize(
        self,
        init_state: list = [1, 1],
        T: float = 100,
        trials: int = 1000,
        common_stock: bool = False,
        opt_type: str = "grid_search",
        config: tuple | dict = (),
        **kwargs,
    ) -> list:
        """Optimizes backtest and strategy and returns best numbers to create the most profit

        NOTE:
        -----
        grid search is faster than simulated annealing with more powerful computers

        :param opt_type: type of optimization (grid search or simulated annealing)
        :type opt_type: str
        :param init_state: inital values for the strat, defaults to [1, 1]
        :type init_state: list, optional
        :param T: temperature value for simulated annealing, defaults to 100
        :type T: float, optional
        :param trials: iterations for simulated annealing, defaults to 1000
        :type trials: int, optional
        :param config: APP_ID, SECRET, USER_AGENT for reddit api
        :type config: tuple or dict
        :param kwargs: the kwargs are strategy specific and is the main item that is going to be
        optimized. ENTER KWARGS IN A RANGE FORMAT AS A LIST Example::

            #[start, stop, step]
            [0,20,2]

        :return: returns a list of the best numbers and the output for those numbers.
            For simulated annealing it will also return the history of how the algorithim
            got to the best outcome
        :rtype: list

        Output
        ------
        ::

            ((State), net worth)
            ((36, 40), 1283666.5067901611)
        """

        opt = Optimize(self.__dict__, Backtest, opt_type=opt_type, **kwargs)

        if common_stock:
            return (
                opt.optimize_(init_state, T, trials),
                opt._find_common_stocks(config),
            )
        return opt.optimize_(init_state, T, trials)

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
            backtest.net_worth.pct_change().std() * math.sqrt(252)
        )
        stats["Sharpe Ratio"] = sharpe

        # ------------  Volatility -----------
        stats[
            "Volatility Annualized (% change)"
        ] = backtest.net_worth.pct_change().std() * math.sqrt(252)

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
            math.sqrt(variance) * backtest.SP500.pct_change().std()
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
