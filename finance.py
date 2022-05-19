# import finance module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


class Finance_Data:
    def __init__(self, ticker, period="MAX"):
        """setup finance data class

        :param ticker: ticker to get data for
        :type ticker: str
        :param period: data period, defaults to "MAX"
        :type period: str, optional
        """
        self.ticker = ticker
        self.complete_data = yf.Ticker(ticker)
        self.data = self.complete_data.history(period=period)
        self.market_data = yf.Ticker("SPY").history(period="MAX").Close

    def percent_return(self, time_frame=None):
        """returns the percent return for the ticker

        :param time_frame: time frame for percent return (needs to be a pandas time frame e.g. "10Y"), defaults to None
        :type time_frame: str, optional
        :return: returns percent return over a time period 
        :rtype: series 
        """
        if time_frame:
            return (self.data.Close.last(time_frame).pct_change() + 1).cumprod()
        return (self.data.Close.pct_change() + 1).cumprod()

    def plot_data(self, plot_type="REGULAR", color="LIGHT"):
        """plots data for ticker

        :param plot_type: can choose between REGULAR, PERCENT and LOG_PERCENT returns for plotting, defaults to "REGULAR"
        :type plot_type: str, optional
        :param color: LIGHT or DARK graph color, defaults to "LIGHT"
        :type color: str, optional
        """

        light_style = "graph_colors/stock-light.mplstyle"
        dark_style = "graph_colors/stock-dark.mplstyle"
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


if __name__ == "__main__":
    pass
