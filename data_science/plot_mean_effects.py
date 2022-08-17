# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker

from data_science import utils


def compute_mean_effects(data_frame, x_column_name, y_column_name, summary_statistic='mean', clip_flag=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    y_vals = data_frame[y_column_name].copy()
    x_vals = data_frame[x_column_name].copy()
    y_vals = utils.replace_invalid(y_vals)
    y_vals = y_vals.astype(float).to_numpy()
    x_vals = utils.replace_invalid(x_vals)

    x_level_vals = data_frame[x_column_name + '_level'].copy().astype(float).to_numpy()

    if clip_flag:
        y_lim = np.percentile(y_vals[np.isfinite(y_vals)], 95)

        idx = y_vals < y_lim
        y_vals = y_vals[idx]
        x_vals = x_vals[idx]
        x_level_vals = x_level_vals[idx]

    if str(x_vals.dtype) in numerics:
        x_vals = x_vals.astype(float).to_numpy()
        idx = np.isfinite(x_vals)
        x_level_vals = x_level_vals[idx]
        x_levels = np.unique(x_level_vals)
        x_levels.sort()
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]

        x = np.zeros(0, dtype=np.float64)
        for l in x_levels:
            idx = x_level_vals == l
            x = np.append(x, np.mean(x_vals[idx]))

        y = np.zeros(0, dtype=np.float64)
        y_full = np.zeros(0, dtype=np.float64)
        support = np.zeros(0, dtype=np.float64)
        variance = np.zeros(0, dtype=np.float64)
        for l in x_levels:
            tmp_y = y_vals[x_level_vals == l]
            tmp_y = tmp_y[np.isfinite(tmp_y)]
            support = np.append(support, tmp_y.size)
            variance = np.append(variance, np.std(tmp_y))

            if type(y_full) == np.ndarray:
                y_full = list()
            y_full.append(np.asarray(tmp_y))

            if summary_statistic is 'mean':
                y = np.append(y, np.mean(tmp_y))
            elif summary_statistic is 'median':
                y = np.append(y, np.median(tmp_y))
            elif summary_statistic is 'none':
                if type(y) == np.ndarray:
                    y = list()
                y.append(np.asarray(tmp_y))
            else:
                raise RuntimeError('Invalid summary statistic: {}'.format(summary_statistic))
    else:
        x_levels = np.unique(x_level_vals)
        idx = np.isfinite(x_levels)
        x_levels = x_levels[idx]
        x_levels.sort()

        x = np.zeros(0, dtype=np.float64)
        y = np.zeros(0, dtype=np.float64)
        y_full = np.zeros(0, dtype=np.float64)
        support = np.zeros(0, dtype=np.float64)
        variance = np.zeros(0, dtype=np.float64)

        for l in x_levels:
            tmp_x = x_vals[x_level_vals == l]
            tmp_x = utils.unique_non_null(tmp_x)
            if len(tmp_x) == 0:
                print('  x col {} is all null'.format(x_column_name))
                # a level combination has no support, so keep a placeholder here
                tmp_x = 'null'
            x = np.append(x, tmp_x)
            tmp_y = y_vals[x_level_vals == l]
            tmp_y = tmp_y[np.isfinite(tmp_y)]
            support = np.append(support, tmp_y.size)
            variance = np.append(variance, np.std(tmp_y))

            if type(y_full) == np.ndarray:
                y_full = list()
            y_full.append(np.asarray(tmp_y))

            if summary_statistic is 'mean':
                y = np.append(y, np.mean(tmp_y))
            elif summary_statistic is 'median':
                y = np.append(y, np.median(tmp_y))
            elif summary_statistic is 'none':
                if type(y) == np.ndarray:
                    y = list()
                y.append(np.asarray(tmp_y))
            else:
                raise RuntimeError('Invalid summary statistic: {}'.format(summary_statistic))

    return x, y_full, y, support, variance


def violin_plotter(x, y, y_summary, support, x_column_name, y_column_name, y_min, y_max):
    ax = plt.gca()

    ax.violinplot(y)
    x_offset = list(range(1, len(x)+1))
    ax.plot(x_offset, y_summary, '_', color='k', markersize=12, linewidth=2)

    for i in range(len(y_summary)):
        s_x = x_offset[i]
        s_y = y_summary[i]

        #ax.text(s_x, s_y, '{0:.3f}'.format(y_summary[i]), horizontalalignment='left', verticalalignment='bottom', fontsize=5, color='k')
        #ax.text(s_x, s_y, ' \u03BC={0:.3f}'.format(y_summary[i]), horizontalalignment='left', verticalalignment='bottom', fontsize=5, color='k')
        ax.text(s_x, s_y, r' $\bar x$={0:.3f}'.format(y_summary[i]), horizontalalignment='left', verticalalignment='bottom', fontsize=8, color='k')

    ax.set_xlabel(x_column_name)
    ax.set_xticks(list(range(1, len(x) + 1)))

    if np.issubdtype(x.dtype, np.number):
        x_labels = [str(round(float(label), 2)) for label in x]
        ax.set_xticklabels(x_labels)
    else:
        ax.set_xticklabels(x)


    ax.set_ylabel(y_column_name)
    # ax.legend(['Mean Value', 'Distribution'])

    #if y_min is not None and y_max is not None:
    plt.ylim([y_min, y_max])

    # ax.set_yscale('log')

    if len(x) >= 4:
        plt.xticks(rotation=45)
    plt.tight_layout()


def box_plotter(x, y, support, x_column_name, y_column_name, y_min, y_max):
    ax = plt.gca()
    ax.boxplot(y)
    ax.set_xlabel(x_column_name)

    if np.issubdtype(x.dtype, np.number):
        x_labels = [str(round(float(label), 2)) for label in x]
        ax.set_xticklabels(x_labels)
    else:
        ax.set_xticklabels(x)

    ax.set_ylabel(y_column_name)

    #if y_min is not None and y_max is not None:
    plt.ylim([y_min, y_max])

    # ax.set_yscale('log')

    if len(x) >= 4:
        plt.xticks(rotation=45)
    plt.tight_layout()


def plotter(x, y, support, x_column_name, y_column_name, y_min, y_max, y_variance):
    ax = plt.gca()
    if y_variance is None:
        ax.plot(x, y, 'o-', markersize=18, linewidth=2)
    else:
        ax.errorbar(x, y, fmt='o-', markersize=18, linewidth=2, yerr=y_variance, elinewidth=1, capsize=12)
    ax.set_xlabel(x_column_name)

    if x.dtype == np.object:
         ax.set_xticklabels(x)
    ax.set_ylabel(y_column_name)

    #if y_min is not None and y_max is not None:
    plt.ylim([y_min, y_max])

    for s_idx in range(len(support)):
        s_x = x[s_idx]
        s_y = y[s_idx]
        ax.text(s_x, s_y, '{}'.format(int(support[s_idx])), horizontalalignment='center', verticalalignment='center', fontsize=6, color='w', weight="bold")

    # ax.set_yscale('log')

    if len(x) >= 4:
        plt.xticks(rotation=45)
    plt.tight_layout()


def main(global_results_csv_filepath, metric, output_dirpath, box_plot_flag, plot_variance_flag, clip_flag, violin_flag, truncate, autoscale):
    ts = box_plot_flag + plot_variance_flag
    if ts > 1:
        raise RuntimeError("Conflicting plot type selections.")

    results_df = pd.read_csv(global_results_csv_filepath)
    results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, (0.3465 + 0.1))

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

    if metric not in list(results_df.columns):
        raise RuntimeError('Selected metric "{}" is not a valid column in the csv file'.format(metric))
    if metric in features_list:
        features_list.remove(metric)

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    y_min = np.inf
    y_max = -np.inf
    x_dict = dict()
    y_dict = dict()
    y_full_dict = dict()
    support_dict = dict()
    var_dict = dict()
    summary_stat = 'mean'
    if box_plot_flag:
        summary_stat = 'none'

    for factor in features_list:
        try:
            print('Computing mean effects for {}'.format(factor))
            x, y_full, y, support, variance = compute_mean_effects(results_df, factor, metric, summary_statistic=summary_stat, clip_flag=clip_flag)
            x_dict[factor] = x
            y_dict[factor] = y
            y_full_dict[factor] = y_full
            support_dict[factor] = support
            var_dict[factor] = variance
            if type(y) == np.ndarray:
                y_min = min(y_min, np.min(y - variance))
                y_max = max(y_max, np.max(y + variance))
            elif type(y) == list:
                for ty in y:
                    y_min = min(y_min, np.min(ty))
                    y_max = max(y_max, np.max(ty))
            else:
                raise RuntimeError('Unexpected y value container type')
        except:
            pass

    # stretch y range by 5%
    delta = y_max - y_min
    y_min -= 0.05 * delta
    y_min = max(0, y_min)
    y_max += 0.05 * delta

    if truncate > 0:
        y_max = truncate

    if autoscale:
        y_min = 0
        y_max = None

    plot_succeeded = False

    if violin_flag:
        fig = plt.figure(figsize=(6, 4.5), dpi=300)
        for factor in features_list:
            print('Plotting violin plot for {}'.format(factor))
            plt.clf()
            try:
                violin_plotter(x_dict[factor], y_full_dict[factor], y_dict[factor], support_dict[factor], factor, metric, y_min, y_max)

                plt.savefig(os.path.join(output_dirpath, '{}.png'.format(factor)))
                plot_succeeded = True
            except:
                print("Factor: {} failed to plot".format(factor))
                pass
        plt.close(fig)

    else:

        fig = plt.figure(figsize=(6, 4.5), dpi=300)
        for factor in features_list:
            print('Plotting mean effects for {}'.format(factor))
            plt.clf()
            try:
                if box_plot_flag:
                    box_plotter(x_dict[factor], y_dict[factor], support_dict[factor], factor, metric, y_min, y_max)
                else:
                    if plot_variance_flag:
                        plotter(x_dict[factor], y_dict[factor], support_dict[factor], factor, metric, y_min, y_max, var_dict[factor])
                    else:
                        plotter(x_dict[factor], y_dict[factor], support_dict[factor], factor, metric, y_min, y_max, None)
                plt.savefig(os.path.join(output_dirpath, '{}.png'.format(factor)))
                plot_succeeded = True
            except:
                print("Factor: {} failed to plot".format(factor))
                pass
        plt.close(fig)

    if not plot_succeeded:
        import shutil
        shutil.rmtree(output_dirpath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot DEX analysis.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--metric', type=str, default='cross_entropy', help='Which column to use as the y-axis')
    parser.add_argument('--box-plot', action='store_true')
    parser.add_argument('--var', action='store_true')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--violin', action='store_true')
    parser.add_argument('--truncate', type=float, default=-1)
    parser.add_argument('--autoscale', action='store_true')
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    main(args.global_results_csv_filepath, args.metric, args.output_dirpath, args.box_plot, args.var, args.clip, args.violin, args.truncate, args.autoscale)

