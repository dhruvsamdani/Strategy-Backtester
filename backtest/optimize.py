import itertools
import random
from multiprocessing import cpu_count, get_context
from typing import Any, Tuple, Type

import numpy as np
from numpy.random import default_rng


class DataMissmatchException(Exception):
    pass


class _Range:
    """Custom range class that accpets floats as step """

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

    def _setup_data(self):
        return [_Range(self.kwargs[i]) for i in list(self.kwargs)]

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

    def opt_func(self, state: list) -> Tuple[list, "Backtest"]:
        """function that needs to be optimized

        :param state: state for the function that is being optimized
        :type state: list
        :return: returns the state and the cost of the state
        :rtype: Tuple
        """
        init_amnt = self.backtest_info["initial_amount"]
        ticker = self.backtest_info["ticker"]
        strat = self.backtest_info["strat"].__class__
        input_data = self.backtest_info["data"]

        for i, k in enumerate(list(self.kwargs)):
            self.kwargs[k] = state[i]

        return (
            state,
            self.backtest(
                init_amnt, ticker, strat, input_data=input_data, **self.kwargs
            )
            .run()
            .net_worth[-1],
        )

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
        return best_state, self.opt_func(best_state), history

    def grid_search(self) -> list[list, float]:
        """finds the optimal numbers for the backtest using a grid search (brute force).
            The algorithm also takes advantage of multiproccesing to speed up the brute force times

        :return: best state and ouptout (cost) of the state
        :rtype: list[list, float]
        """
        expanded_indicator = [i._range() for i in self.indicators]
        input_list = list(itertools.product(*expanded_indicator))

        pool = get_context("fork").Pool(cpu_count())
        res = pool.map(self.opt_func, input_list)
        pool.close()

        return max(res, key=lambda x: x[1])

