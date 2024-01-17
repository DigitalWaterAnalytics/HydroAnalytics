import os
from typing import Union, List, Tuple, Dict
from datetime import datetime, time
import json
import io

import pandas as pd
import unittest
import requests
import numpy as np
import matplotlib.pyplot as plt
import scipy


def event_short_name(
        event_date: Union[datetime, pd.Timestamp],
        date_time_format: str = '%m/%d/%Y %H:%M:%S',
        output_date_time_format: str = '%Y%b%dT%H',
        output_prefix: str = '',
        output_suffix: str = '',
) -> str:
    """
    This function returns a short name for the event
    :param event_date: The date of the event
    :param date_time_format:  The format of the event date
    :param output_date_time_format: The format of the output date
    :param output_prefix: The prefix to use for the output
    :param output_suffix: The suffix to use for the output
    :return: A string with the event name
    """

    if isinstance(event_date, str):
        event_date = datetime.strptime(event_date, date_time_format)

    event_name = rf'{output_prefix}{event_date.strftime(output_date_time_format)}{output_suffix}'

    return event_name


def post_event_time(events: pd.DataFrame, ts_end: pd.Timestamp) -> pd.DataFrame:
    """
    This function returns a dataframe of the times between the end of an event and the start of a new one
    :param events: A dataframe with the event attributes
    :param ts_end: The end of the timeseries data
    :return: A dataframe with the timeseries data post event
    """
    post_times = events['start'].shift(-1) - events['actual_end']
    return post_times.fillna(ts_end - events.iloc[-1]['end'])


def antecedent_event_time(events: pd.DataFrame, ts_start: pd.Timestamp) -> pd.DataFrame:
    """
    This function returns a dataframe of the antecedent times before an event
    :param events:  A dataframe with the event attributes
    :param ts_start:  The start of the timeseries data
    :return: A dataframe with the timeseries data antecedent to the event
    """
    antecedent_times = events['start'] - events['actual_end'].shift(1)
    return antecedent_times.fillna(events.iloc[0]['start'] - ts_start)


def get_event_attributes(
        latitude: float,
        longitude: float,
        events: pd.DataFrame,
        units: str = 'english',
        series: str = 'pds'):
    """
    This function returns a dataframe of the events statistical attributes compared to long term records
    :param latitude: Latitude of the location of interest
    :param longitude: Longitude of the location of interest
    :param events: Pandas dataframe of events
    :param units: Units of to use for the event attributes. Valid values are 'english' and 'metric'
    :param series: Series to use for the event attributes. Valid values are Partial Duration 'pds' and
    Annual Maximum Series 'ams'
    :return: A dataframe with the event attributes
    """

    NOAA_HDSC_REST_URL = r'https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/'

    query_params = {
        'lat': latitude,
        'lon': longitude,
        'data': 'depth',
        'units': units,
        'series': series,
    }

    response = requests.get(f'{NOAA_HDSC_REST_URL}fe_text_mean.csv', params=query_params, verify=False)
    content = response.content.decode('utf-8')
    results = pd.read_csv(io.StringIO(content), skiprows=13, skipfooter=3, index_col=0)

    durations = []

    for duration in results.index:
        duration = duration.replace(':', '')
        duration = duration.replace('-', '')
        if 'min' in duration:
            duration = float(duration.replace('min', ''))
        elif 'hr' in duration:
            duration = float(duration.replace('hr', '')) * 60.0
        elif 'day' in duration:
            duration = float(duration.replace('day', '')) * 60.0 * 24.0

        durations.append(duration)

    durations = np.array(durations)
    results.index = durations
    result_columns = results.columns.to_list()
    results['0'] = 0.0
    results = results[['0'] + result_columns]
    return_periods = np.array([float(c) for c in results.columns])

    interp = scipy.interpolate.interp2d(return_periods, durations, results.values)

    events_cp = events.copy()
    events_cp['return_period'] = None

    for event_index, event_row in events_cp.iterrows():
        event_duration = event_row['duration'].total_seconds() / 60.0
        event_precip_total = event_row['precip_total']
        event_durations = np.array([event_duration])
        precip_totals = interp(return_periods, event_durations)
        return_period = np.interp(event_precip_total, precip_totals, return_periods)
        events_cp.loc[event_index, 'return_period'] = return_period

    return events_cp


def get_events(
        rainfall: pd.DataFrame,
        inter_event_time: pd.Timedelta = pd.Timedelta('24 hour'),
        floor: Union[float, Dict[str, float]] = 0.01,
        flow_or_depth: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    This function returns a dataframe of events from a rainfall dataframe. An event is defined as a period of time
    :param rainfall: A dataframe with a datetime index and multiple columns representing incremental rainfall
    accumulations in the area of interest
    :param inter_event_time:
    :param floor: Threshold for the minimum rainfall depth or flow for an event to be considered
    :param flow_or_depth:
    :return: A data frame with the event attributes
    """

    data = rainfall.copy()

    if flow_or_depth is not None:
        data = pd.concat(objs=[data, flow_or_depth], axis='columns')

    if isinstance(floor, dict):
        floor = pd.Series(floor)
        floor = floor.reindex(data.columns)
        floor = floor.fillna(0.0)
    else:
        floor = pd.Series([floor] * len(data.columns), index=data.columns)

    data = data - floor
    rolling_sum = data.rolling(inter_event_time).sum()
    rolling_sum_combined = rolling_sum.sum(axis='columns')
    max_of_fields = rolling_sum_combined.max(axis="rows")

    normalized_rolling_sum = rolling_sum_combined / max_of_fields
    normalized_rolling_sum[normalized_rolling_sum.index[0]] = 0
    normalized_rolling_sum[normalized_rolling_sum.index[-1]] = 0
    normalized_rolling_sum_index = (normalized_rolling_sum <= 0.0).astype(int)
    normalized_rolling_sum_index_diff = normalized_rolling_sum_index.diff()

    events = pd.DataFrame(
        {
            "start": normalized_rolling_sum_index_diff[normalized_rolling_sum_index_diff == -1].index,
            "end": normalized_rolling_sum_index_diff[normalized_rolling_sum_index_diff == 1].index,
        }
    )

    ts_start, ts_end = data.index[0], data.index[-1]

    events['name'] = events.apply(
        lambda row: event_short_name(row['start'], output_prefix='', output_suffix=''),
        axis='columns'
    )

    events = events[['name', 'start', 'end']]

    def get_actual_end(row, ts):
        sub_ts = ts[row['start']:row['end']]

        return sub_ts.loc[sub_ts[sub_ts.columns[0]] > 0].dropna().index[-1]

    events['actual_end'] = events.apply(func=get_actual_end, axis='columns', args=(rainfall,))

    events['duration'] = events['actual_end'] - events['start']
    events['post_event_time'] = post_event_time(events, ts_end)
    events['antecedent_event_time'] = antecedent_event_time(events, ts_start)

    def get_rainfall_peak_intensity(row, ts):
        return ts[row['start']:row['end']].max()

    def get_rainfall_sum(row, ts):
        return ts[row['start']:row['end']].sum()

    events['precip_peak'] = events.apply(func=get_rainfall_peak_intensity, axis='columns', args=(rainfall,))
    events['precip_total'] = events.apply(func=get_rainfall_sum, axis='columns', args=(rainfall,))

    return events


def write_swmm_rainfall_file(rainfall: pd.DataFrame, events: pd.DataFrame, output_file_path: str):
    """
    This function writes a SWMM rainfall file for each event
    :param rainfall:
    :param events:
    :param output_file_path:
    :return:
    """
    for _, event_row in events.iterrows():
        event_name = f'{event_row["name"]}'
        filename = os.path.join(output_file_path, f'{event_name}.dat')
        ts = rainfall[event_row['start']:event_row['end']]
        write_swmm_rainfall_file(ts, event_name, filename)


def cluster_events(events: pd.DataFrame, cluster_columns: List[str], number_of_clusters: int) -> pd.DataFrame:
    """
    This function clusters events based on the specified columns
    :param events: A dataframe with the event attributes
    :param cluster_columns: A list of columns to use for clustering
    :param number_of_clusters: The number of clusters to create
    :return: A dataframe with the event attributes and cluster labels
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    cluster_data = events[cluster_columns].copy()
    cluster_data = MinMaxScaler().fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(cluster_data)
    cluster_labels = kmeans.predict(cluster_data)
    events['cluster'] = cluster_labels
    return events
