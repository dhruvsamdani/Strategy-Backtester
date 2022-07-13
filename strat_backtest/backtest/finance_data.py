import glob
from importlib import resources
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


class Finance_Data:
    market_data = yf.Ticker("SPY").history(period="MAX").Close
    risk_free_rate_10y = yf.Ticker("^TNX").history(period="7d").Close[-1] / 100
    risk_free_rate_90d = yf.Ticker("^IRX").history(period="7d").Close[-1] / 100

    def __init__(self, ticker: str = None, period: str = "MAX"):
        """setup finance data class
        :param ticker: ticker to get data for
        :type ticker: str
        :param period: data period, defaults to "MAX"
        :type period: str, optional
        """

        self.ticker = ticker
        self.complete_data = yf.Ticker(ticker)
        self.data = self.complete_data.history(period=period)

    def percent_return(self, time_frame: str = None) -> pd.Series:
        """returns the percent return for the ticker

        :param time_frame: time frame for percent return (needs to be a pandas time frame e.g. "10Y"), defaults to None
        :type time_frame: str, optional
        :return: returns percent return over a time period
        :rtype: pd.Series
        """
        if time_frame:
            return (self.data.Close.last(time_frame).pct_change() + 1).cumprod()
        return (self.data.Close.pct_change() + 1).cumprod()

    def plot_data(self, plot_type: str = "REGULAR", color: str = "LIGHT"):
        """plots data for ticker

        :param plot_type: can choose between REGULAR, PERCENT and LOG_PERCENT returns for plotting, defaults to "REGULAR"
        :type plot_type: str, optional
        :param color: LIGHT or DARK graph color, defaults to "LIGHT"
        :type color: str, optional
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
        if plot_type == "REGULAR":
            ax = self.data.Close.plot()
        elif plot_type == "PERCENT":
            ax = self.percent_return().plot(title=f"Percent Return of {self.ticker}")
        elif plot_type == "LOG_PERCENT":
            ax = (
                np.log(self.data.Close.pct_change() + 1)
                .cumsum()
                .plot(title=f"Log Percent Return of {self.ticker}")
            )
        plt.text(
            0.5,
            0.5,
            self.ticker,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=36,
            weight="bold",
            alpha=0.3,
            color=text_color,
            variant="small-caps",
            zorder=1,
        )
        plt.show()


def load_data(path: str) -> dict:
    """loads finanacial data from a path

    :param path: path to data csv or dir with data csv
    :type path: str
    :return: dataframe of financial data
    :rtype: dict
    """
    file_list = glob.glob(f"{path}/*.csv") if os.path.isdir(path) else [path]
    return {
        Path(data).stem: pd.read_csv(data, index_col=0, parse_dates=True)
        for data in file_list
    }


def download_data(*args, period: str = "10Y", **kwargs) -> pd.DataFrame:
    """retrieves financial data from yfinance api to be downloaded

    *args: tickers that need data to be retrieved
    :param period: period for the data
    :type period: str
    **kwargs: any settings that the yfinance api needs

    :return: dataframe of ticker data
    :rtype: DataFrame

    Usage
    ::

    downdolad_data('AAPL', 'MSFT', period='max')

    """
    tickers = " ".join(args)
    return yf.download(tickers, auto_adjust=True, group_by="ticker", **kwargs).last(
        period
    )
