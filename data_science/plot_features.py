# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn

from data_science import utils


def plot_two_columns(ax, results_df, x_column_name, y_column_name, y_axis_logscale=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    y_vals = results_df[y_column_name].copy()
    x_vals = results_df[x_column_name].copy()
    y_vals = utils.replace_invalid(y_vals)
    y_vals = y_vals.astype(float).to_numpy()
    y_lim = np.percentile(y_vals[np.isfinite(y_vals)], 95)
    x_vals = utils.replace_invalid(x_vals)

    idx = y_vals < y_lim
    y_vals = y_vals[idx]
    x_vals = x_vals[idx]

    if str(x_vals.dtype) in numerics:
        x_vals = x_vals.astype(float).to_numpy()
        if y_axis_logscale:
            ax.set_yscale('log')
        alpha = 500 * (1.0 / float(len(y_vals)))
        ax.scatter(x_vals, y_vals, c='b', s=4, alpha=alpha)
        # ax.hexbin(x_vals, y_vals, gridsize=100)

        ax.set_xlabel(x_column_name)
        ax.set_ylabel(y_column_name)
        # plt.ylim(0, y_lim)
        # ax.set_title(x_column_name)
    else:
        # categories = x_vals.unique()
        categories = utils.unique_non_null(x_vals)
        x = list()
        for c in categories:
            vals = y_vals[x_vals == c]
            vals = vals[np.isfinite(vals)]
            x.append(vals)
        if y_axis_logscale:
            ax.set_yscale('log')
        idx = 0
        categories = categories.tolist()
        while idx < len(categories):
            if len(x[idx]) == 0:
                del categories[idx]
                del x[idx]
            else:
                idx += 1
        ax.boxplot(x)
        # ax.violinplot(x)
        ax.set_xlabel(x_column_name)
        ax.set_xticklabels(categories)
        ax.set_ylabel(y_column_name)
        # plt.ylim(0, y_lim)
        plt.xticks(rotation=45)
        # if len(categories) >= 3:
        #     plt.xticks(rotation=45)
        # ax.set_title(x_column_name)
    plt.tight_layout()


def main(global_results_csv, metric, output_dir, log_scale):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_df = pd.read_csv(global_results_csv)
    results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, (0.3465 + 0.1))
    results_df['ground_truth'] = results_df['ground_truth'].astype('category')

    # TODO update this with the new level information

    to_drop = [fn for fn in list(results_df.columns) if fn.endswith('_level')]
    results_df = results_df.drop(columns=to_drop)
    results_df.reset_index(drop=True, inplace=True)

    # results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, 0.4)

    # drop columns with only one unique value
    to_drop = list()
    for col in list(results_df.columns):
        if len(results_df[col].unique()) <= 1:
            to_drop.append(col)
    to_drop.append('model_name')
    to_drop.append('team_name')
    results_df = results_df.drop(columns=to_drop)
    results_df.reset_index(drop=True, inplace=True)

    features_list = list(results_df.columns)
    if metric not in features_list:
        raise RuntimeError('Selected metric "{}" is not a valid column in the csv file'.format(metric))
    features_list.remove(metric)
    fig = plt.figure(figsize=(6, 4.5), dpi=300)
    for name in features_list:
        plt.clf()
        ax = plt.gca()
        try:
            plot_two_columns(ax, results_df, name, metric, y_axis_logscale=log_scale)
            plt.savefig(os.path.join(output_dir, '{}.png'.format(name)))
        except:
            print("Factor: {} failed to plot".format(name))
            pass
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot each column of the global results csv file.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--metric', type=str, default='cross_entropy', help='Which column to use as the y-axis')
    parser.add_argument('--output-dirpath', type=str, required=True)
    parser.add_argument('--log-scale', action='store_true')

    args = parser.parse_args()
    main(args.global_results_csv_filepath, args.metric, args.output_dirpath, args.log_scale)

