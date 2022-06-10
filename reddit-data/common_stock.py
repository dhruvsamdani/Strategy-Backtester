import configparser
import re
from collections import Counter
from ftplib import FTP
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import praw
import requests

"""
TODO:
    Documentation
    Ticker blacklist (IPO, USD, etc)
    Code formatting
    Integration with backtest
"""


def get_all_tickers():
    if Path("tickers.csv").exists():
        return pd.read_csv("./tickers.csv").symbol

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
    listed = pd.read_table(StringIO(second_nas), sep="|")[["symbol", "security name"]]

    most_c = pd.read_table("./most_common.txt", header=None)
    most_c = most_c[most_c[0].str.len() <= 4]

    tickers = listed.merge(traded, how="left")
    tickers = tickers[
        ~tickers.symbol.str.contains(r"\.|\$", na=True)
        & ((tickers.symbol.str.len() > 1))
    ]
    tickers = tickers[~tickers.symbol.isin(most_c[0])]
    tickers.to_csv("tickers.csv")
    return tickers.symbol


def clean_text(text: str, ticker: bool = False):
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
    if ticker:
        regex_pattern = re.compile("|".join(pattern))
        reg_filter = r"([A-Z]{2,5})|(\$[A-z]+)"
        text = re.sub(regex_pattern, "", text)
        return ["".join(x) for x in re.findall(reg_filter, text)]

    text = text.lower()
    punct = r"""(amp)?[,'"#$%&;:()[\]{}=_`~\/\^\*\-]"""
    pattern.append(punct)

    regex_pattern = re.compile("|".join(pattern))
    text = re.sub(regex_pattern, "", text)
    text = re.sub(r"(\n)|(\t)", " ", text)
    return text


def enter_data(subreddit):
    config = configparser.ConfigParser()
    config.read(str(Path().absolute().parent / "config.ini"))
    APP_ID = config["REDDIT"]["API_KEY"]
    SECRET = config["REDDIT"]["SECRET"]
    USER_AGENT = config["REDDIT"]["USER_AGENT"]

    reddit = praw.Reddit(client_id=APP_ID, client_secret=SECRET, user_agent=USER_AGENT)

    red = reddit.subreddit(subreddit).top(time_filter="week", limit=20)

    comments = lambda sid: requests.get(
        rf"https://www.reddit.com/r/{sid}/comments.json",
        headers={"User-Agent": USER_AGENT},
    ).json()["data"]["children"]

    return pd.DataFrame(
        (
            [
                clean_text(submission.title, ticker=True),
                clean_text(submission.selftext, ticker=True),
                (
                    [
                        clean_text(
                            data["data"]["body"] if "body" in data["data"] else "",
                            ticker=True,
                        )
                        for data in comments(submission)
                    ]
                ),
            ]
            for submission in red
        )
    )


def combine(row):

    if row == []:
        return []
    else:
        return np.hstack(row)


def most_common(n: int, subreddit: str):
    ticker_info = enter_data(subreddit)
    tickers = set(get_all_tickers())
    ticker_info = ticker_info.sum(axis=1).apply(
        lambda row: [] if row == [] else np.hstack(row)
    )
    ticker_info = ticker_info.apply(
        lambda row: [item for item in row if item.lower() in tickers]
    )
    return Counter(np.hstack(ticker_info)).most_common(n)


if __name__ == "__main__":
    a = most_common(10, "stocks")
    print(a)
