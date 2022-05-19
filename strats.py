import os

import matplotlib.pyplot as plt
import pandas as pd

from finance import Finance_Data


class Strategey:
    def __init__(self, initial_amount, ticker):
        """Base class for any strategy to be backtested

        :param initial_amount: initial amount of money to start with 
        :type initial_amount: int 
        :param ticker: ticker symbol 
        :type ticker: str 
        """

        self.initial_amount = initial_amount
        self.ticker = Finance_Data(ticker)
        self.data = self.ticker.data
        self.backtest = self.setup_backtest()

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

    def setup_backtest(self):
        """Creates backtest DataFrame

        :return: empty backtest DataFrame 
        :rtype: DataFrame 
        """
        backtest = pd.DataFrame(
            columns=["net_worth", "buy_shares", "sell_shares", "shares_owned"]
        )
        return backtest

    def calc_trading_shares(self, indicators=[], stock_to_buy=1, stock_to_sell=1):
        """Marks when to buy and sell a stock 

        :param indicators: indicators to be passed to the strategy (e.g moving average), defaults to []
        :type indicators: list, optional
        :param stock_to_buy: how many stocks to buy when buy signal passed, defaults to 1
        :type stock_to_buy: int, optional
        :param stock_to_sell: how many stocks to sell when sell signal passed, defaults to 1
        :type stock_to_sell: int, optional
        """
        self.buy_and_sell(indicators)
        self.backtest.fillna(0, inplace=True)
        self.backtest.loc[self.backtest.buy == True, "buy_shares"] = stock_to_buy
        self.backtest.loc[self.backtest.sell == True, "sell_shares"] = stock_to_sell
        self.backtest["shares_owned"] = (
            self.backtest.buy_shares - self.backtest.sell_shares
        ).cumsum()

    def calculate_net_worth(self):
        """Calculates net worth from the strategy

        :return: daily networth from the time period of the strategey 
        :rtype: list 
        """
        buy = self.backtest.buy_shares
        sell = self.backtest.sell_shares
        shares_owned = self.backtest.shares_owned

        net_worth = [self.initial_amount]
        current_net_worth = self.initial_amount
        buy_cost = 0

        real_price = self.data.Close

        for i in range(1, len(buy)):

            buy_cost += buy[i] * real_price[i]

            if sell[i] != 0:
                current_net_worth += sell[i] * real_price[i]

            net_worth.append(
                current_net_worth + real_price[i] * shares_owned[i] - buy_cost
            )
        return net_worth

    def run_backtest(self, indicators=[]):
        """Runs the backtest and fills out backtest DataFrame 

        :param indicators: indicators to be passed to the strategy (e.g moving average), defaults to []
        :type indicators: list, optional
        :return: backtest data 
        :rtype: DataFrame 
        """
        self.calc_trading_shares(indicators)
        self.backtest["net_worth"] = self.calculate_net_worth()
        self.backtest = pd.concat(
            [
                self.backtest,
                pd.DataFrame(
                    {"SP500": self.ticker.market_data.tail(len(self.ticker.data))}
                ),
            ],
            axis=1,
        ).fillna(0)
        return self.backtest

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
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.text(
            0.5,
            0.5,
            self.ticker.ticker,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=76,
            weight="bold",
            alpha=0.3,
            color=text_color,
            variant="small-caps",
            zorder=1,
        )
        if not os.path.isdir("./Graphs"):
            os.makedirs("Graphs")
        plt.savefig("Graphs/" + filename)


class MA_Cross_Strat(Strategey):
    def __init__(self, initial_amount, ticker):
        """Strategy that buys stock when a short moving average crosses a long one and sells when the long crosses the short.
        20 day crossing 100 and vise versa. Child class of Strategy

        :param initial_amount: inital amount of money 
        :type initial_amount: int 
        :param ticker: ticker symbol 
        :type ticker: str 
        """
        Strategey.__init__(self, initial_amount, ticker)
        self.configure_data()

    def configure_data(self):
        """Configures and sets up the data. Adds indicators that strategey needs to the DataFrame for the ticker
        """
        data = self.data
        data["percent-return"] = self.ticker.percent_return(time_frame="10Y")
        data["twenty_day"] = self.rolling_average(data["percent-return"], 20)
        data["hundred_day"] = self.rolling_average(data["percent-return"], 100)

    def buy_and_sell(self, indicators=[]):
        """Main strategey is in this function. Decides when to mark a day as buy and sell based on the indicators

        :param indicators: indicators that strategey is dependent on, defaults to []
        :type indicators: list, required 
        """
        twenty_ma = indicators[0]
        hundred_ma = indicators[1]

        self.backtest["buy"] = (twenty_ma > hundred_ma) & (
            (twenty_ma.shift(1) < hundred_ma.shift(1))
            & (self.initial_amount > self.data.Close)
        )

        buy_indicator = self.backtest.buy[self.backtest.buy == True]
        if buy_indicator.any():
            first_buy = buy_indicator.index[0]
            last_buy = buy_indicator.index[-1]
        else:
            first_buy = self.backtest.index[0]
            last_buy = self.backtest.index[0]

        self.backtest["sell"] = (
            (twenty_ma < hundred_ma)
            & (twenty_ma.shift(1) > hundred_ma.shift(1))
            & (self.backtest.index > first_buy)
        )

        self.backtest.loc[
            (self.backtest.cumsum().diff(axis=1).sell != 0)
            & (self.backtest.index > last_buy),
            "sell",
        ] = False


if __name__ == "__main__":

    # Test example for msft with a $300 start amount
    strat = MA_Cross_Strat(300, "MSFT")
    indicators = [strat.data.twenty_day, strat.data.hundred_day]
    backtest = strat.run_backtest(indicators=indicators)

    strat.plot_data(
        (
            (backtest[["net_worth", "SP500"]].last("10Y").pct_change() + 1).cumprod()
            * 100
        )
        - 100,
        title="Percent return of MA strategy against time",
        ylabel="Percent Returns",
        color="DARK",
    )
