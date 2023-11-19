# python imports
from enum import Flag, auto

# third party imports
import pandas as pd


# local imports

class Seasonality(Flag):
    """
    Enum for seasonality options
    """
    HourOfDay = auto(), "Hour of Day"
    DayOfWeek = auto(), "Day of Week"
    DayOfYear = auto(), "Day of Year"


def add_seasonality(ts: pd.DataFrame, seasonality_flags: Seasonality) -> pd.DataFrame:
    """
    Add seasonality to a time series
    :param ts:
    :param seasonality_flags:
    :return:
    """
    if Seasonality.HourOfDay in seasonality_flags:
        ts['hour_of_day'] = ts.index.hour
    if Seasonality.DayOfWeek in seasonality_flags:
        ts['day_of_week'] = ts.index.dayofweek
    if Seasonality.DayOfYear in seasonality_flags:
        ts['day_of_year'] = ts.index.dayofyear

    return ts


def low_pass_filter(ts: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Low pass filter a time series
    :param ts:
    :param window:
    :return:
    """
    ts['low_pass'] = ts['precip'].rolling(window=window, center=True).mean()

    return ts