import pytest
from backtest import Backtest, load_data
from backtest.custom_strats import MA_Cross_Strat


@pytest.mark.parametrize(
    "fast, lagging, net_worth",
    [(36, 40, 1283666.449897766), (40, 100, 61864.069396972656)],
)
def test_cross_strat(fast, lagging, net_worth):
    data = load_data("./data/aapl.csv")["aapl"].last("10Y")
    data.columns = data.columns.str.lower()

    def run_backtest():
        return (
            Backtest(
                5000,
                "AAPL",
                MA_Cross_Strat,
                input_data=data,
                fast=fast,
                lagging=lagging,
            )
            .run()
            .net_worth[-1]
        )

    assert run_backtest() == net_worth
