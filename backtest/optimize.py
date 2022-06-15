import itertools
import random
from multiprocessing import cpu_count, get_context
from typing import Any, List, Tuple, Type

import numpy as np
from numpy.random import default_rng
from reddit_data.common_stock import Reddit_Stocks

from . import download_data


class DataMissmatchException(Exception):
    pass


class NoOptException(Exception):
    pass


class _Range:
    """Custom range class that accpets floats as step"""

    def __init__(self, lst=[]):
        if len(lst) != 3:
            raise DataMissmatchException(
                "The list should contain 3 items a start, stop (inclusive, exclusive) and step"
            )
        self.start, self.stop, self.step = lst

    def _range(self):
        return np.arange(self.start, self.stop, self.step)


class Optimize:
    def __init__(self, backtest_info: dict, backtest: Type["Backtest"], **kwargs):
        """Initialize the optimize class and load default informatio

        :param backtest_info: information that the backtest needs to run
        :type backtest_info: dict
        :param backtest: a backtest class to run the backtest on
        :type backtest: Type[&quot;Backtest&quot;]
        """
        self.kwargs = kwargs
        self.backtest_info = backtest_info
        self.backtest = backtest
        self.indicators = self._setup_data()
        self.best_state = None
        self.opt = None

    def _setup_data(self):
        return [_Range(self.kwargs[i]) for i in list(self.kwargs)]

    def _find_common_stocks(self) -> List[Tuple]:
        """Finds the most commonly talked about stocks and optimizes trading strategy on them

        :raises NoOptException: function can only be run after original optimization has been run
        :return: returns optimized parameters for common stocks
        :rtype: List[Tuple]
        """

        if self.opt is None:
            raise NoOptException("There needs to be an opt function to use")

        c_stocks = Reddit_Stocks(
            1, ["stocks", "wallstreetbets", "finance", "StockMarket", "investing"]
        ).most_common()
        most_common_stocks = []

        tickers = download_data(" ".join([c[0] for c in c_stocks]))
        opt_func = self.opt

        for stock, _ in c_stocks:
            params = {
                "ticker": stock,
                "input_data": tickers[stock] if len(c_stocks) > 1 else tickers,
            }
            most_common_stocks.append((stock, opt_func(stock, **params)))

        return most_common_stocks

    def _neighborhood(self, state: Any, amplitude: int) -> list:
        """neighborhood for SA

        :param state: initial state
        :type state: Any
        :param amplitude: amplitude for noise (higher creates more noise)
        :type amplitude: int
        :return: returns a new point
        :rtype: list
        """

        rng = default_rng()
        size = len(self.indicators)
        next_s = lambda cur_state: cur_state + rng.integers(
            -1, 2, size=size
        ) * rng.integers(-amplitude, amplitude + 1, size=size) * [
            indicator.step for indicator in self.indicators
        ]

        new_state = next_s(state)
        while not (new_state > 0).all() or np.array_equal(new_state, state):
            new_state = next_s(state)

        for i in range(len(state)):
            if self.indicators[i].start > new_state[i]:
                new_state[i] = self.indicators[i].start
            if self.indicators[i].stop < new_state[i]:
                new_state[i] = self.indicators[i].stop
        return new_state

    def opt_func(
        self,
        state: list,
        ticker: str = None,
        input_data: Type["pd.DataFrame"] = None,
    ) -> Tuple[list, "Backtest"]:

        """function that needs to be optimized, takes in ticker and input_data to find common stocks

        :param state: state for the function that is being optimized
        :type state: list
        :param ticker: ticker for stock
        :type ticker: str
        :param input_data: OHLCV data for stock
        :type input_data: pd.DataFrame
        :return: returns the state and the cost of the state
        :rtype: Tuple
        """
        init_amnt = self.backtest_info["initial_amount"]
        ticker = self.backtest_info["ticker"] if ticker is None else ticker
        strat = self.backtest_info["strat"].__class__
        input_data_backtest = (
            self.backtest_info["data"] if input_data is None else input_data
        )

        for i, k in enumerate(list(self.kwargs)):
            self.kwargs[k] = state[i]

        return (
            state,
            self.backtest(
                init_amnt, ticker, strat, input_data=input_data_backtest, **self.kwargs
            )
            .run()
            .net_worth[-1],
        )

    #! CURRENTLY DOES NOT WORK FOR FIND COMMON STOCKS
    def simulated_annealing(
        self, init_state: Any, T: float, iterations: int
    ) -> Tuple[list, list]:
        """Optimized function based on simulated annealing

        :param init_state: initial state provided
        :type init_state: Any
        :param T: inital temperature
        :type T: float
        :param iterations: number of iterations
        :type iterations: int
        :return: best state and history of how it got there
        :rtype: Tuple[list, list]
        """
        self.opt = (self.simulated_annealing, init_state, iterations)
        state = best_state = init_state
        history = [init_state]
        temp = T
        cur_cost = self.opt_func(state)[1]
        for _ in reversed(range(iterations)):
            next_state = self._neighborhood(state, 10)
            new_cost = self.opt_func(next_state)[1]
            delta_cost = new_cost - cur_cost

            if delta_cost > 0:
                state = next_state
                if new_cost > cur_cost:
                    best_state = next_state
                cur_cost = self.opt_func(state)[1]
            elif np.exp(delta_cost / temp) > random.uniform(0, 1):
                state = next_state
                cur_cost = self.opt_func(state)[1]
            history.append(state)
            temp *= 0.8

        history.append(best_state)
        self.best_state = best_state
        return best_state, self.opt_func(best_state), history

    def grid_search(self, **kwargs) -> list[list, float]:
        """finds the optimal numbers for the backtest using a grid search (brute force).
            The algorithm also takes advantage of multiproccesing to speed up the brute force times

        **kwargs: key word arguments for the opt function

        :return: best state and ouptout (cost) of the state
        :rtype: list[list, float]
        """
        self.opt = self.grid_search
        expanded_indicator = [i._range() for i in self.indicators]
        input_list = list(itertools.product(*expanded_indicator))

        with get_context("fork").Pool(cpu_count()) as pool:
            res = [
                pool.apply_async(
                    self.opt_func,
                    (state,),
                    kwargs,
                ).get()
                for state in input_list
            ]

        best_state = max(res, key=lambda x: x[1])
        self.best_state = best_state
        return best_state
