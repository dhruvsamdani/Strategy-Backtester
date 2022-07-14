import configparser
from importlib import resources
import re
from collections import Counter
from dataclasses import dataclass
from ftplib import FTP
from functools import reduce
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import praw
import requests
from strat_backtest.config import CONFIG_PATH, ROOT_PATH

"""
SAMPLE CONFIG INI
[REDDIT]
API_KEY = <api key (app id)>
SECRET = <secret>
USER_AGENT = <user agent>
"""


@dataclass(frozen=True)
class RedditConfig:

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    """CONFIG WILL BE FAVORED OVER PARAMETERS """
    APP_ID: str = None
    SECRET: str = None
    USER_AGENT: str = None

    if config.has_section("REDDIT"):
        APP_ID = config["REDDIT"]["API_KEY"]
        SECRET = config["REDDIT"]["SECRET"]
        USER_AGENT = config["REDDIT"]["USER_AGENT"]


class Reddit_Stocks:
    def __init__(
        self, number: int, subreddit: List[str], config: tuple | dict = ()
    ) -> None:
        self.n = number
        self.subreddits = subreddit
        self.red_config = (
            RedditConfig(**config)
            if isinstance(config, dict)
            else RedditConfig(*config)
        )
        self.reddit = praw.Reddit(
            client_id=self.red_config.APP_ID,
            client_secret=self.red_config.SECRET,
            user_agent=self.red_config.USER_AGENT,
        )

    def _get_all_tickers(self) -> set:
        """Gets all tickers from nasdaq and removes the most commonly talked about words
        Ex: to, the, etc.

        :return: set of tickers
        :rtype: set
        """

        with resources.path("strat_backtest.reddit_data", "tickers.csv") as tickers:
            if Path(tickers).exists():
                return set(pd.read_csv(tickers).symbol)

        traded = BytesIO()
        listed = BytesIO()
        with FTP("ftp.nasdaqtrader.com") as ftp:
            ftp.login()
            ftp.retrbinary("RETR /SymbolDirectory/nasdaqlisted.txt", traded.write)
            ftp.retrbinary("RETR /SymbolDirectory/nasdaqtraded.txt", listed.write)

        traded.seek(0)
        listed.seek(0)

        nasdaq_listed = traded.read().decode().lower()
        second_nas = listed.read().decode().lower()

        traded = pd.read_table(StringIO(nasdaq_listed), sep="|")[
            ["symbol", "security name"]
        ]
        listed = pd.read_table(StringIO(second_nas), sep="|")[
            ["symbol", "security name"]
        ]

        with resources.path("strat_backtest.reddit_data", "most_common.txt") as mc:
            most_c = pd.read_table(mc, header=None)
        most_c = most_c[most_c[0].str.len() <= 4]

        tickers = listed.merge(traded, how="left")
        tickers = tickers[
            ~tickers.symbol.str.contains(r"\.|\$", na=True)
            & ((tickers.symbol.str.len() > 1))
        ]
        tickers = tickers[~tickers.symbol.isin(most_c[0])]
        tickers.to_csv(ROOT_PATH / r"reddit_data\tickers.csv")
        return set(tickers.symbol)

    def _clean_text(self, text: str) -> List[str]:
        """cleans text to extract only capital words and words starting with $

        :param text: text to clean
        :type text: str
        :return: list of cleaned text
        :rtype: list
        """
        # emoticons symbols & pictographs transport & map symbols flags (iOS)
        pattern = [
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,7})([\/\w#?=%+&;.-]*)",
        ]
        regex_pattern = re.compile("|".join(pattern))
        reg_filter = r"([A-Z]{2,5})|\$([A-z]+)"
        text = re.sub(regex_pattern, "", text)
        return ["".join(x) for x in re.findall(reg_filter, text)]

    def _enter_data(self, subreddit: str) -> pd.DataFrame:
        """Cleans the data for the title, text, and comments for a subreddit

        :param subreddit: the subredit to take data from
        :type subreddit: str
        :return: DataFrame for cleaned text
        :rtype: pd.DataFrame
        """

        red = self.reddit.subreddit(subreddit).top(time_filter="week", limit=20)

        comments = lambda sid: requests.get(
            rf"https://www.reddit.com/r/{sid}/comments.json",
            headers={"User-Agent": self.red_config.USER_AGENT},
        ).json()["data"]["children"]

        return pd.DataFrame(
            (
                [
                    self._clean_text(submission.title),
                    self._clean_text(submission.selftext),
                    (
                        [
                            self._clean_text(
                                data["data"]["body"] if "body" in data["data"] else ""
                            )
                            for data in comments(submission)
                        ]
                    ),
                ]
                for submission in red
            )
        )

    def most_common(self) -> List[Tuple]:
        """Returns the most common tickers and stock symbols for all the data

        :return: list of most common tickers
        :rtype: list
        """
        tickers = self._get_all_tickers()
        blacklist = {
            "dcf",
            "dtc",
            "usd",
            "dd",
            "cpi",
            "fomo",
            "sec",
            "ipo",
            "usd",
            "esg",
            "tv",
        }
        tickers_df = reduce(
            lambda a, b: a.add(b), [self._enter_data(subr) for subr in self.subreddits]
        )
        tickers_df = tickers_df.sum(axis=1).apply(
            lambda row: []
            if row == [] or row == 0
            else [
                item
                for item in np.hstack(row)
                if item.lower() in tickers and item.lower() not in blacklist
            ]
        )
        return Counter(np.hstack(tickers_df)).most_common(self.n)


if __name__ == "__main__":
    rs = Reddit_Stocks(
        10, ["stocks", "wallstreetbets", "finance", "StockMarket", "investing"]
    )
    print(rs.most_common())
