import numpy as np
import pandas as pd


def replace_invalid(df):
    df.fillna(value=np.nan, inplace=True)
    try:
        df.replace(to_replace=[None], value=np.nan, inplace=True)
    except:
        pass
    try:
        df.replace(to_replace='None', value=np.nan, inplace=True)
    except:
        pass
    return df


def unique_non_null(s):
    return np.asarray(s.dropna().unique())


def filter_dataframe_by_cross_entropy_threshold(data_frame, ce_threshold):
    metric = 'cross_entropy'
    primary_key = 'team_name'
    secondary_key = 'execution_time_stamp'

    teams = dict()

    primary_key_vals = data_frame[primary_key].unique()
    for pk in primary_key_vals:
        primary_df = data_frame[data_frame[primary_key] == pk]
        secondary_key_vals = primary_df[secondary_key].unique()
        for sk in secondary_key_vals:
            run_df = data_frame[data_frame[secondary_key] == sk]
            team_name = run_df[primary_key].unique()[0]
            col = run_df[metric]
            ce = col.mean(axis=0)
            if ce < ce_threshold:
                if not team_name in teams.keys():
                    teams[team_name] = list()
                teams[team_name].append(run_df[secondary_key].unique()[0])

    subset_df = None
    for team in teams.keys():
        primary_df = data_frame[data_frame[primary_key] == team]
        for timestamp in teams[team]:
            print('{} - {} '.format(team, timestamp))
            run_df = primary_df[primary_df[secondary_key] == timestamp]
            if subset_df is None:
                subset_df = run_df
            else:
                subset_df = subset_df.append(run_df)

    return subset_df