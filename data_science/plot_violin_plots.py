# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_science import utils


def plot_two_columns(ax, results_df, x_column_name, y_column_name, y_axis_logscale=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    y_vals = results_df[y_column_name].copy()
    x_vals = results_df[x_column_name].copy()
    y_vals = utils.replace_invalid(y_vals)
    y_vals = y_vals.astype(float).to_numpy()
    x_vals = utils.replace_invalid(x_vals)

    if str(x_vals.dtype) in numerics:
        x_vals = x_vals.astype(float).to_numpy()
        if y_axis_logscale:
            ax.set_yscale('log')
        ax.scatter(x_vals, y_vals, c='b', s=48, alpha=0.1)
        ax.set_xlabel(x_column_name)
        ax.set_ylabel(y_column_name)
        # ax.set_title(x_column_name)
    else:
        categories = list(x_vals.unique())
        x = list()
        for c in categories:
            if isinstance(c, (float, complex)) and np.isnan(c):
                vals = y_vals[x_vals.isnull()]
            else:
                vals = y_vals[x_vals == c]
            vals = vals[np.isfinite(vals)]
            x.append(vals)

        for i in range(len(categories)):
            if isinstance(categories[i], (float, complex)):
                if np.isnan(categories[i]):
                    categories[i] = 'None'
        order_idx = np.argsort(categories)
        categories = [categories[i] for i in order_idx]
        x = [x[i] for i in order_idx]

        if y_axis_logscale:
            ax.set_yscale('log')

        # for i in range(len(categories)):
        #     idx = np.random.rand(len(x[i]))
        #     idx = ((idx - 0.5) / 2.0) + (i+1)
        #     plt.scatter(idx, x[i], c='b', s=48, alpha=0.05)

        try:
            ax.violinplot(x)
        except:
            ax.boxplot(x)
        ax.set_xlabel(x_column_name)
        plt.xticks(list(range(1, len(categories)+1)), list(categories))
        ax.set_ylabel(y_column_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # ax.set_title(x_column_name)


def main(global_results_csv_filepath, metric, output_dirpath):
    results_df = pd.read_csv(global_results_csv_filepath)
    # treat two boolean columns categorically
    results_df['ground_truth'] = results_df['ground_truth'].astype('category')

    # results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, 0.45)

    # modify dataframe to null out certain nonsensical data
    # idx = results_df['ground_truth'] == 0
    # results_df['number_triggered_classes'][idx] = np.nan
    # results_df['triggered_fraction'][idx] = np.nan

    # drop columns with only one unique value
    to_drop = list()
    for col in list(results_df.columns):
        if len(results_df[col].unique()) <= 1:
            to_drop.append(col)
    to_drop.append('model_name')
    to_drop.append('execution_time_stamp')
    to_drop.append('team_name')
    results_df = results_df.drop(columns=to_drop)

    columns_list = list(results_df.columns)
    features_list = list()
    for column in columns_list:
        if column.endswith('_level'): continue
        level_column_name = column + "_level"
        if level_column_name in columns_list:
            features_list.append(column)

    features_list.append('ground_truth')

    if metric not in list(results_df.columns):
        raise RuntimeError('Selected metric "{}" is not a valid column in the csv file'.format(metric))
    if metric in features_list:
        features_list.remove(metric)

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # fig = plt.figure(figsize=(16, 9), dpi=200)
    fig = plt.figure(figsize=(8, 6), dpi=400)
    for factor in features_list:
        plt.clf()
        plot_two_columns(plt.gca(), results_df, factor, metric)
        plt.savefig(os.path.join(output_dirpath, '{}.png'.format(factor)))
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot DEX analysis.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--metric', type=str, default='cross_entropy', help='Which column to use as the y-axis')
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    main(args.global_results_csv_filepath, args.metric, args.output_dirpath)

