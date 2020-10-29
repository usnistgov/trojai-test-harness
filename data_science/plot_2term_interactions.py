# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_science import utils


def get_subplot_size(n):
    h = 1
    w = 1
    i = 0
    while w * h < n:
        if i % 2 == 0:
            w = w + 1
            i = i + 1
        else:
            h = h + 1
            i = i + 1
    return w, h


def compute_mean_effects_matrix(data_frame, factor1_name, factor2_name, metric_name, summary_statistic='mean'):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    f1_is_numeric = str(data_frame[factor1_name].dtype) in numerics
    f2_is_numeric = str(data_frame[factor2_name].dtype) in numerics

    # determine the clusters
    f1_levels = utils.unique_non_null(data_frame[factor1_name + '_level'])
    if f1_is_numeric:
        f1_vals_unique = np.zeros(0, dtype=np.float32)
        for l in f1_levels:
            key = factor1_name + '_level'
            idx = data_frame[key] == l
            f1_df = data_frame[idx]
            f1_vals = f1_df[factor1_name]
            f1_vals = f1_vals.to_numpy().astype(np.float32)
            f1_vals = f1_vals[np.isfinite(f1_vals)]
            f1_vals_unique = np.append(f1_vals_unique, np.mean(f1_vals).round(decimals=2))
    else:
        f1_vals_unique = list()
        for l in f1_levels:
            key = factor1_name + '_level'
            idx = data_frame[key] == l
            f1_df = data_frame[idx]
            f1_vals = f1_df[factor1_name]
            f1_vals_unique.append(utils.unique_non_null(f1_vals)[0])
        f1_vals_unique = np.asarray(f1_vals_unique)

    f2_levels = utils.unique_non_null(data_frame[factor2_name + '_level'])
    if f2_is_numeric:
        f2_vals_unique = np.zeros(0, dtype=np.float32)
        for l in f2_levels:
            key = factor2_name + '_level'
            idx = data_frame[key] == l
            f2_df = data_frame[idx]
            f2_vals = f2_df[factor2_name]
            f2_vals = f2_vals.to_numpy().astype(np.float32)
            f2_vals = f2_vals[np.isfinite(f2_vals)]
            f2_vals_unique = np.append(f2_vals_unique, np.mean(f2_vals).round(decimals=2))
    else:
        f2_vals_unique = list()
        for l in f2_levels:
            key = factor2_name + '_level'
            idx = data_frame[key] == l
            f2_df = data_frame[idx]
            f2_vals = f2_df[factor2_name]
            f2_vals_unique.append(utils.unique_non_null(f2_vals)[0])
        f2_vals_unique = np.asarray(f2_vals_unique)

    means_matrix = None
    support_matrix = None
    for f1_level in f1_levels:
        key = factor1_name + '_level'
        idx1 = data_frame[key] == f1_level
        f1_df = data_frame[idx1]

        y = np.zeros(0, dtype=np.float64)
        y_counts = np.zeros(0, dtype=np.float64)

        for f2_level in f2_levels:
            key = factor2_name + '_level'
            idx2 = f1_df[key] == f2_level
            f2_df = f1_df[idx2]

            metric_vals = f2_df[metric_name].to_numpy().astype(np.float32)
            if np.size(metric_vals) == 0:
                # no support for this factor configuration
                y = np.append(y, np.nan)
                y_counts = np.append(y_counts, 0)
            else:
                if summary_statistic is 'mean':
                    y = np.append(y, np.mean(metric_vals))
                    y_counts = np.append(y_counts, metric_vals.size)
                elif summary_statistic is 'median':
                    y = np.append(y, np.median(metric_vals))
                    y_counts = np.append(y_counts, metric_vals.size)
                else:
                    raise RuntimeError('Invalid summary statistic: {}'.format(summary_statistic))
        if means_matrix is None:
            means_matrix = y
            support_matrix = y_counts
        else:
            means_matrix = np.vstack((means_matrix, y))
            support_matrix = np.vstack((support_matrix, y_counts))

    return means_matrix, support_matrix, f1_vals_unique, f2_vals_unique


def main(factor_global_results_csv_filepath, output_dirpath):

    results_df = pd.read_csv(factor_global_results_csv_filepath)
    # treat two boolean columns categorically
    results_df['ground_truth'] = results_df['ground_truth'].astype('category')

    columns = list(results_df.columns)
    factors_list = list()
    for c in columns:
        if c.endswith('_level'):
            results_df[c] = results_df[c].astype('category')
            factors_list.append(c.replace('_level', ''))

    results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, 0.4)

    metric = "cross_entropy"
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # TODO continue updating this to use the new factor csv file

    y_min = np.inf
    y_max = -np.inf
    matrix_dict = dict()
    support_dict = dict()
    f_vals = dict()

    for factor1 in factors_list:
        matrix_dict[factor1] = dict()
        support_dict[factor1] = dict()
        for factor2 in factors_list:
            if factor1 == factor2:
                continue
            mean_effects_matrix, support_matrix, f1_vals, f2_vals = compute_mean_effects_matrix(results_df, factor1, factor2, metric, summary_statistic='mean')
            matrix_dict[factor1][factor2] = mean_effects_matrix
            support_dict[factor1][factor2] = support_matrix
            f_vals[factor1] = f1_vals
            tmp = mean_effects_matrix[np.isfinite(mean_effects_matrix)]
            if tmp.size != 0:  # some factor combinations are nonsense
                y_min = min(y_min, np.min(tmp))
                y_max = max(y_max, np.max(tmp))

    # stretch y range by 10%
    delta = y_max - y_min
    y_min -= 0.1 * delta
    y_max += 0.1 * delta


    # plot factor1 levels as a series of vertical subplots and the factor2 levels as a series of horizontal subplots

    for factor1 in factors_list:
        for factor2 in factors_list:
            if factor1 == factor2:
                continue
            plt.clf()
            mean_effects_matrix = matrix_dict[factor1][factor2]
            support_matrix = support_dict[factor1][factor2]
            x_vals = f_vals[factor1]
            plot_titles = f_vals[factor2].astype(np.str)
            n_w, n_h = get_subplot_size(mean_effects_matrix.shape[1])

            # fig, ax_list = plt.subplots(1, n_w, figsize=(4 * n_w, 4), dpi=100)
            fig, ax_list = plt.subplots(n_h, n_w, figsize=(4 * n_w, 4 * n_h), dpi=200)
            ax_list = ax_list.reshape(-1)  # make axis list 1d
            rotate_labels = False
            for i in range(mean_effects_matrix.shape[1]):
                ax = ax_list[i]
                x = x_vals.reshape(-1)
                y = mean_effects_matrix[:, i].reshape(-1)
                support = support_matrix[:, i].reshape(-1)

                ax.plot(x, y, 'o-', markersize=18, linewidth=2)
                for s_idx in range(len(support)):
                    s_x = x[s_idx]
                    s_y = y[s_idx]
                    ax.text(s_x, s_y, '{}'.format(int(support[s_idx])), horizontalalignment='center', verticalalignment='center', fontsize=6, color='w')
                ax.set_xlabel(factor1)
                if x.dtype == np.object:
                    ax.set_xticklabels(x)
                ax.set_ylabel(metric)
                ax.set_ylim([y_min, y_max])
                ax.set_title('{} : {}'.format(factor2, plot_titles[i]))

                if len(x) >= 4:
                    rotate_labels = True

            # hide the unused plots
            for i in range(mean_effects_matrix.shape[1], n_w * n_h):
                ax_list[i].axis('off')
            if rotate_labels:
                for ax in ax_list:
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(45)
            plt.tight_layout()

            plt.savefig(os.path.join(output_dirpath, '{}-{}.png'.format(factor1, factor2)))
            plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot DEX analysis.')
    parser.add_argument('--factor-global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    main(args.factor_global_results_csv_filepath, args.output_dirpath)

