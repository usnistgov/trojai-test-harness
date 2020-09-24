# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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