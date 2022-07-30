"""Contains methods and classes to collect data from
https://www.cryptodatadownload.com.
"""

import ssl

import numpy as np
import pandas as pd
import yfinance as yf

import requests 
import warnings
import gzip
import os
import math

from datetime import datetime
from dateutil import rrule
from datetime import datetime


ssl._create_default_https_context = ssl._create_unverified_context


class YahooDataDownload:
    """Provides methods for retrieving data on different cryptocurrencies from
    https://www.cryptodatadownload.com/cdd/.

    Attributes
    ----------
    url : str
        The url for collecting data from YahooDataDownload.

    Methods
    -------
    fetch(exchange_name,base_symbol,quote_symbol,timeframe,include_all_volumes=False)
        Fetches data for different exchanges and cryptocurrency pairs.

    """

    def __init__(self) -> None:
        self.url = "https://www.cryptodatadownload.com/cdd/"

    def fetch_default(self,
                      exchange_name: str,
                      base_symbol: str,
                      quote_symbol: str,
                      timeframe: str,
                      include_all_volumes: bool = False) -> pd.DataFrame:
        """Fetches data from all exchanges that match the evaluation structure.

        Parameters
        ----------
        exchange_name : str
            The name of the exchange.
        base_symbol : str
            The base symbol fo the cryptocurrency pair.
        quote_symbol : str
            The quote symbol fo the cryptocurrency pair.
        timeframe : {"d", "h", "m"}
            The timeframe to collect data from.
        include_all_volumes : bool, optional
            Whether or not to include both base and quote volume.

        Returns
        -------
        `pd.DataFrame`
            A open, high, low, close and volume for the specified exchange and
            cryptocurrency pair.
        """

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        warnings.filterwarnings("ignore")
        pd.options.display.float_format = '{:.4%}'.format

        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        
        filename = "{}_{}{}_{}.csv".format(exchange_name, quote_symbol, base_symbol, timeframe)
        base_vc = "Volume {}".format(base_symbol)
        new_base_vc = "volume_base"
        quote_vc = "Volume {}".format(quote_symbol)
        new_quote_vc = "volume_quote"
       
        start = '2016-01-01'
        end = datetime.today().strftime('%Y-%m-%d')

        df = yf.download(base_symbol, start = start, end = end)
        df['asset'] = base_symbol
        df.ffill(axis = 0)
        
        #df = pd.read_csv(self.url + filename, skiprows=1)
        #df = df[::-1]
        #df = df.drop(["symbol"], axis=1)
        #df = df.rename({base_vc: new_base_vc, quote_vc: new_quote_vc, "Date": "date"}, axis=1)

        #if "d" in timeframe:
        #    df["date"] = pd.to_datetime(df["date"])
        #elif "h" in timeframe:
        #    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")

        #df = df.set_index("date")
        df.columns = [name.lower() for name in df.columns]
        #df = df.reset_index()
        #if not include_all_volumes:
        #    df = df.drop([new_quote_vc], axis=1)
        #    df = df.rename({new_base_vc: "volume"}, axis=1)
        #    return df
        return df

   
    def fetch(self,
              exchange_name: str,
              base_symbol: str,
              quote_symbol: str,
              timeframe: str,
              include_all_volumes: bool = False) -> pd.DataFrame:
        """Fetches data for different exchanges and cryptocurrency pairs.

        Parameters
        ----------
        exchange_name : str
            The name of the exchange.
        base_symbol : str
            The base symbol fo the cryptocurrency pair.
        quote_symbol : str
            The quote symbol fo the cryptocurrency pair.
        timeframe : {"d", "h", "m"}
            The timeframe to collect data from.
        include_all_volumes : bool, optional
            Whether or not to include both base and quote volume.

        Returns
        -------
        `pd.DataFrame`
            A open, high, low, close and volume for the specified exchange and
            cryptocurrency pair.
        """

        return self.fetch_default(exchange_name,
                                  base_symbol,
                                  quote_symbol,
                                  timeframe,
                                  include_all_volumes=include_all_volumes)
