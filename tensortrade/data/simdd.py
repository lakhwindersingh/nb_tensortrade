"""Contains methods and classes to collect data from
https://www.cryptodatadownload.com.
"""

import ssl

import numpy as np
import pandas as pd

import requests 
import warnings
import gzip
import os
import math
import random

from datetime import datetime, date
from dateutil import rrule
from datetime import datetime
import tensortrade.stochastic.processes as sp 
from dateutil import relativedelta
from datetime import timedelta

ssl._create_default_https_context = ssl._create_unverified_context


class SimulatedDataDownload:
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
        start_date = date(2020, 1, 1)
        tod = date.today()

        if "d" in timeframe:
            num_days = int(timeframe.replace("d",""))
            delta = tod - datetime.timedelta(days=num_days)
            start_date = delta.strftime('%Y-%m-%d')
     
        data = sp.cox(
            base_price=1000,
            base_volume=5,
            start_date=start_date,
            times_to_generate=num_days,
            time_frame=timeframe
        )

#        data = sp.heston(
#            base_price=1000,
#            base_volume=5,
#            start_date=start_date,
#            times_to_generate=num_days,
#            time_frame=timeframe
#        )

        x = random.randint(1,1000)
        data = data*random.uniform(x-x/10, x+x/10)
        return data


   
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
