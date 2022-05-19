# Strategy Backtester

Strategy Backtester is a **python** program that backtests various strategies to make money on the stock market

**Example Plot**:
![Example Strategy Plot](./Graphs/data.png)

## How it Works:

The strategey backtester works by pulling data from the yahoo finance api (unoffical) and manipulates that data to create indicators. For example the only indicators present in the code right now are 2 moving average indicators.

The data is then entered into pandas DataFrames and then a strategy can be made with a manipulation of the DataFrame data. After the strategy is created it is automatically backtested and then the data can be plotted onto a graph for easy access

## Dependencies:

- **Python 3.0+**
- [Numpy](https://github.com/numpy/numpy)
- [Pandas](https://github.com/pandas-dev/pandas)
- [Yahoo Finance](https://github.com/ranaroussi/yfinance)

```bash
pip install numpy
pip install pandas
pip install yfinance
```

## Installation:

Download the `finance.py` and `strats.py` to the folder where the backtest is going to be run. Once all the dependencies are met the program can be implemented correctly[^1]

[^1]: If you want to plot the data make sure to also download the `graph_colors` folder which contains the customizations for the graphs

## Usage:

```python
from strats import MA_Cross_Strat

# setup ticker and initial amount
strat = MA_Cross_Strat(300, "MSFT")

# setup indicators
indicators = [strat.data.twenty_day, strat.data.hundred_day]

# simulate backtest
backtest = strat.run_backtest(indicators=indicators)

data = ((backtest[["net_worth", "SP500"]].last("10Y").pct_change() + 1).cumprod() * 100) - 100

# plot data with customizations
# more customizations in docstring for plot_data
strat.plot_data(
    data,
    title="Percent return of MA strategy against time",
    ylabel="Percent Returns",
    color="DARK",
)

# Graph will be stored in ./Graphs
```

## Work in Progress

1. Add more strategies
2. Adapt code to work with **options**
3. Make script to analyze other sources of data to get better insight into which stocks to backtest
4. **Maybe:** _Add algotrading bot to program_
