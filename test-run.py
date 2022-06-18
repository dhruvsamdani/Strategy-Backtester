from backtest import Backtest, download_data, load_data
from backtest.custom_strats import MA_Cross_Strat, Ten_Percent_Strat

# Download data for tickers
# download_data("AAPL", "MSFT", "TSLA").AAPL.to_csv("./data/aapl.csv")
# download_data("GME").to_csv("./data/gme.csv")

# Load data from a directory
data = load_data("./data/aapl.csv")["aapl"].last("10Y")
data.columns = data.columns.str.lower()

# # Setup and Run backtest
backtest = Backtest(5000, "AAPL", MA_Cross_Strat, input_data=data, fast=36, lagging=40)

output = backtest.run()

opt = backtest.optimize(
    init_state=[10, 60],
    fast=[36, 42, 2],
    lagging=[40, 210, 10],
    opt_type="grid_search",
    common_stock=True,
)
# Outputting optimized solution
print(opt)


# # Stats
# backtest.metrics(output=True)

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
