import pytest
from strat_backtest.backtest import Backtest, load_data
from strat_backtest.backtest.custom_strats import MA_Cross_Strat
from importlib import resources

with resources.path("strat_backtest.data", "aapl.csv") as aapl:
    TEST_AAPL_DATA = load_data(aapl)["aapl"].last("10Y")
    TEST_AAPL_DATA.columns = TEST_AAPL_DATA.columns.str.lower()


@pytest.mark.parametrize(
    "fast, lagging, net_worth",
    [(36, 40, 1283666.449897766), (40, 100, 61864.069396972656)],
)
def test_cross_strat(fast, lagging, net_worth):
    def run_backtest():
        return (
            Backtest(
                5000,
                "AAPL",
                MA_Cross_Strat,
                input_data=TEST_AAPL_DATA,
                fast=fast,
                lagging=lagging,
            )
            .run()
            .net_worth[-1]
        )

    assert run_backtest() == net_worth
