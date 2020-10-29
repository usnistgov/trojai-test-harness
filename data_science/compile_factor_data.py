# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from data_science import utils


def main(global_results_csv_filepath, output_dirpath, num_levels):

    results_df = pd.read_csv(global_results_csv_filepath)
    # treat two boolean columns categorically
    results_df['trigger_target_class'] = results_df['trigger_target_class'].astype('category')
    results_df['ground_truth'] = results_df['ground_truth'].astype('category')
    results_df['poisoned'] = results_df['poisoned'].astype('category')

    results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, 0.4)

    # modify dataframe to remove out certain nonsensical data
    idx = results_df['ground_truth'] == 0
    results_df['number_triggered_classes'][idx] = np.nan
    results_df['triggered_fraction'][idx] = np.nan

    # split trigger_type_option into two columns
    trigger_type = results_df['trigger_type']
    trigger_type_option = results_df['trigger_type_option']
    polygon_side_count = trigger_type_option.copy()
    instagram_filter_type = trigger_type_option.copy()
    polygon_side_count[trigger_type == 'instagram'] = np.nan
    instagram_filter_type[trigger_type == 'polygon'] = np.nan

    results_df.drop(columns=['trigger_type_option'])
    results_df['polygon_side_count'] = polygon_side_count
    results_df['polygon_side_count'] = results_df['polygon_side_count'].astype('float32')
    results_df['instagram_filter_type'] = instagram_filter_type
    results_df = results_df.drop(columns=['trigger_type_option'])

    # drop columns with only one unique value, since they cannot meaningfully influence the trojan detector
    to_drop = list()
    for col in list(results_df.columns):
        if len(results_df[col].unique()) <= 1:
            to_drop.append(col)
    # prevent the team name or execution time stamp columns from being removed
    if 'team_name' in to_drop:
        to_drop.remove('team_name')
    if 'execution_time_stamp' in to_drop:
        to_drop.remove('execution_time_stamp')
    # remove nonsensical columns
    to_drop.append('poisoned')
    to_drop.append('master_seed')
    to_drop.append('triggered_classes')
    to_drop.append('trigger_color')
    to_drop.append('final_clean_data_n_total')
    to_drop.append('final_triggered_data_n_total')
    to_drop.append('optimizer_0')
    to_drop.append('final_train_loss')
    to_drop.append('final_train_acc')
    to_drop.append('final_combined_val_acc')
    to_drop.append('final_combined_val_loss')
    to_drop.append('final_clean_val_acc')
    to_drop.append('final_triggered_val_acc')
    to_drop.append('final_clean_data_test_acc')
    to_drop.append('final_triggered_data_test_acc')
    to_drop.append('final_example_acc')
    to_drop.append('trigger_target_class')
    to_drop.append('trigger_behavior')
    to_drop.append('foreground_size_percentage_of_image_min')
    to_drop.append('foreground_size_percentage_of_image_max')
    to_drop.append('foreground_size_pixels_min')
    to_drop.append('foreground_size_pixels_max')
    to_drop.append('trigger_size_percentage_of_foreground_min')
    to_drop.append('trigger_size_percentage_of_foreground_max')
    to_drop.append('final_clean_val_loss')
    to_drop.append('final_triggered_val_loss')
    to_drop.append('training_wall_time_sec')
    to_drop.append('test_wall_time_sec')

    results_df = results_df.drop(columns=to_drop)
    results_df.reset_index(drop=True, inplace=True)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    factor_df = results_df[['team_name','execution_time_stamp', 'ground_truth', 'predicted', 'cross_entropy', 'model_name']].copy()
    columns = list(results_df.columns)
    for c in factor_df.columns:
        columns.remove(c)
    for col_name in columns:
        factor_col_name = col_name + '_level'
        col = results_df[col_name]

        if str(col.dtype) in numerics:
            vals = col.to_numpy().astype(np.float32)
            non_nan_idx = np.isfinite(vals)
            nan_idx = np.logical_not(non_nan_idx)
            subset_vals = vals[np.isfinite(vals)]

            kmeans = KMeans(num_levels).fit(subset_vals.reshape(-1, 1))
            clusters = kmeans.predict(subset_vals.reshape(-1, 1))

            vals[nan_idx] = np.nan
            vals[non_nan_idx] = clusters
        else:
            vals = col.copy().to_numpy()
            vals_unique = utils.unique_non_null(col)
            for i in range(len(vals_unique)):
                vals[vals == vals_unique[i]] = i

        factor_df[col_name] = col
        factor_df[factor_col_name] = vals

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    factor_df.reset_index(drop=True, inplace=True)
    output_filepath = os.path.join(output_dirpath, 'factor-global-results.csv')
    factor_df.to_csv(output_filepath, na_rep='nan', index=False)


if __name__ == "__main__":
    import argparse

    # TODO this script will need to be modified for each round

    parser = argparse.ArgumentParser(description='Script to convert the global results csv into a table for use in mean effects and two factor interaction analysis.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--num-levels', type=int, default=2)
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    main(args.global_results_csv_filepath, args.output_dirpath, args.num_levels)

