import itertools
import random
from multiprocessing import cpu_count, get_context
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

import numpy as np
from numpy.random import default_rng

from strat_backtest.reddit_data import Reddit_Stocks

from strat_backtest.backtest.finance_data import download_data

if TYPE_CHECKING:
    import pandas as pd

    from strats import Backtest


class DataMismatchException(Exception):
    pass


class NoOptException(Exception):
    pass


class _Range:
    """Custom range class that accepts floats as step"""

    def __init__(self, lst: List = []):
        if len(lst) != 3:
            raise DataMismatchException(
                "The list should contain 3 items a start, stop (inclusive, exclusive) and step"
            )
        self.start, self.stop, self.step = lst

    def _range(self):
        return np.arange(self.start, self.stop, self.step)


class Optimize:
    def __init__(
        self,
        backtest_info: dict,
        backtest: Type["Backtest"],
        opt_type: str = "grid_search",
        sa_data: Optional[dict] = None,
        **kwargs
    ):
        """Initialize the optimize class and load default information

        :param backtest_info: information that the backtest needs to run
        :type backtest_info: dict
        :param backtest: a backtest class to run the backtest on
        :type backtest: Type["Backtest";]
        """
        self.kwargs = kwargs
        self.backtest_info = backtest_info
        self.backtest = backtest
        self.indicators = self._setup_data()
        self.best_state = None
        self.opt_type = opt_type

    def _setup_data(self) -> List[_Range]:
        return [_Range(self.kwargs[i]) for i in list(self.kwargs)]

    def _find_common_stocks(self, config: dict = {}) -> List[Tuple]:
        """Finds the most commonly talked about stocks and optimizes trading strategy on them

        :raises NoOptException: function can only be run after original optimization has been run
        :return: returns optimized parameters for common stocks
        :rtype: List[Tuple]
        """

        c_stocks = Reddit_Stocks(
            10,
            ["stocks", "wallstreetbets", "finance", "StockMarket", "investing"],
            config,
        ).most_common()

        most_common_stocks = []

        tickers = download_data(" ".join([c[0] for c in c_stocks]))

        for stock, _ in c_stocks:
            args = []
            if self.opt_type != "grid_search":
                args = self.sa_data
                if self.sa_data == None:
                    raise NoOptException(
                        "You must run a simulated annealing opt before this one"
                    )
            params = {
                "ticker": stock,
                "input_data": tickers[stock] if len(c_stocks) > 1 else tickers,
            }
            most_common_stocks.append((stock, self.optimize_(*args, **params)))

        return most_common_stocks

    def opt_func(
        self,
        state: list,
        ticker: Optional[str] = None,
        input_data: Optional["pd.DataFrame"] = None,
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

    def _simulated_annealing(
        self, init_state: Any, T: float, iterations: int, **kwargs
    ) -> Tuple[list, list]:
        """Optimized function based on simulated annealing

        :param init_state: initial state provided
        :type init_state: Any
        :param T: initial temperature
        :type T: float
        :param iterations: number of iterations
        :type iterations: int
        :return: best state and history of how it got there
        :rtype: Tuple[list, list]
        """

        self.sa_data = {tuple(init_state), T, iterations}

        def _neighborhood(state: Any, amplitude: int) -> List[float]:
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

        state = best_state = init_state
        history = [init_state]
        temp = T
        cur_cost = self.opt_func(state)[1]
        for _ in reversed(range(iterations)):
            next_state = _neighborhood(state, 10)
            new_cost = self.opt_func(next_state, **kwargs)[1]
            delta_cost = new_cost - cur_cost

            if delta_cost > 0:
                state = next_state
                if new_cost > cur_cost:
                    best_state = next_state
                cur_cost = self.opt_func(state, **kwargs)[1]
            elif np.exp(delta_cost / temp) > random.uniform(0, 1):
                state = next_state
                cur_cost = self.opt_func(state, **kwargs)[1]
            history.append(state)
            temp *= 0.8

        history.append(best_state)
        self.best_state = best_state
        return self.opt_func(best_state)

    def _grid_search(self, **kwargs) -> Tuple[list, float]:
        """finds the optimal numbers for the backtest using a grid search (brute force).
            The algorithm also takes advantage of multiprocessing to speed up the brute force times

        **kwargs: key word arguments for the opt function

        :return: best state and output (cost) of the state
        :rtype: list[list, float]
        """
        expanded_indicator = [i._range() for i in self.indicators]
        input_list = list(itertools.product(*expanded_indicator))

        with get_context("fork").Pool(cpu_count()) as pool:
            res = [
                pool.apply_async(self.opt_func, (state,), kwargs).get()
                for state in input_list
            ]

        best_state = max(res, key=lambda x: x[1])
        self.best_state = best_state
        return best_state

    def optimize_(self, *args, **kwargs) -> Tuple[tuple, float]:
        """Optimization function

        *args: arguments for simulated annealing
        **kwargs: internal use for finding and optimizing common stocks

        :return: best state and net worth from state
        :rtype: tuple
        """
        if self.opt_type == "grid_search":
            return self._grid_search(**kwargs)
        else:
            return self._simulated_annealing(*args, **kwargs)
