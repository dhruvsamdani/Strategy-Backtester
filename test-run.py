from turtle import back

from custom_strats import *
from finance_data import download_data, load_data
from strats import Backtest

# Download data for tickers
# download_data('AAPL', 'MSFT', 'TSLA').AAPL.to_csv("./data/aapl.csv")

# Load data from a directory
data = load_data("./data")

# Setup and Run backtest
backtest = Backtest(5000, "AAPL", MA_Cross_Strat, input_data=data["aapl"].last("10Y"))
output = backtest.run()
backtest.metrics()

# Stats
# backtest.metrics(output=True)

# Orders
# orders = backtest.strat.orders
# orders.to_df().to_csv("orders.csv")

backtest.strat.plot_data(
    ((output[["net_worth", "SP500"]].last("10Y").pct_change() + 1).cumprod() * 100)
    - 100,
    title="Percent return of Crossover strategy against time",
    ylabel="Percent Returns",
    color="LIGHT",
)
