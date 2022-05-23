from finance_data import download_data, load_data
from strats import Backtest, MA_Cross_Strat

# Download data for tickers
# download_data('AAPL', 'MSFT', 'TSLA').AAPL.to_csv("./data/aapl.csv")

# Load data from a directory
data = load_data("./data")

# Enter data into strategy
crossover = MA_Cross_Strat("aapl", data["aapl"])

# Initalize backtest
backtest = Backtest(100, "aapl", input_data=data["aapl"])
backtest.setup_strat(crossover)

output = backtest.run()

# Stats
# backtest.metrics(output=False)

# Orders
# orders = backtest.strat.orders
# orders.to_df().to_csv("orders.csv")

crossover.plot_data(
    ((output[["net_worth", "SP500"]].last("10Y").pct_change() + 1).cumprod() * 100)
    - 100,
    title="Percent return of MA strategy against time",
    ylabel="Percent Returns",
    color="LIGHT",
)
