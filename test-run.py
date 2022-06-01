from backtest import Backtest, load_data
from backtest.custom_strats import MA_Cross_Strat, Ten_Percent_Strat

# Download data for tickers
# download_data('AAPL', 'MSFT', 'TSLA').AAPL.to_csv("./data/aapl.csv")
# download_data("AMZN").AMZN.to_csv("./data/aapl.csv")


# Load data from a directory
data = load_data("./data/aapl.csv")["aapl"].last("10Y")
data.columns = data.columns.str.lower()

# # Setup and Run backtest
backtest = Backtest(5000, "AAPL", MA_Cross_Strat, input_data=data, fast=20, lagging=100)

output = backtest.run()

# Optimizing the backtest
opt = backtest.optimize(
    "grid_search", init_state=[1.05, 0.99], fast=[10, 42, 2], lagging=[40, 210, 10]
)

# Outputting optimized solution
print(opt)


# # Stats
# # backtest.metrics(output=True)

# # Orders
# # orders = backtest.strat.orders
# # orders.to_df().to_csv("orders.csv")

# Plotting Data
# backtest.strat.plot_data(
#     ((output[["net_worth", "SP500"]].last("10Y").pct_change() + 1).cumprod() * 100)
#     - 100,
#     title="Percent return of Crossover strategy against time",
#     ylabel="Percent Returns",
#     color="LIGHT",
# )

