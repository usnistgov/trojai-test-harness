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
    return w, h\


def plot_two_columns(ax, results_df, x_column_name, y_column_name, y_axis_logscale=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    y_vals = results_df[y_column_name].copy()
    x_vals = results_df[x_column_name].copy()
    y_vals = utils.replace_invalid(y_vals)
    y_vals = y_vals.astype(float).to_numpy()
    x_vals = utils.replace_invalid(x_vals)

    # clip the y-values
    # y_vals = np.clip(y_vals, 1e-2, np.inf)

    if str(x_vals.dtype) in numerics:
        x_vals = x_vals.astype(float).to_numpy()
        if y_axis_logscale:
            ax.set_yscale('log')
        ax.scatter(x_vals, y_vals, c='b', s=2)
        ax.set_xlabel(x_column_name)
        ax.set_ylabel(y_column_name)
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
        ax.boxplot(x)
        ax.set_xlabel(x_column_name)
        ax.set_xticklabels(categories)
        ax.set_ylabel(y_column_name)
        if len(categories) > 10:
            plt.xticks(rotation=45)
        # ax.set_title(x_column_name)


def plot_feature_list(data_frame, column_list, metric_name):
    n_w, n_h = get_subplot_size(len(column_list))

    fig, ax_list = plt.subplots(n_h, n_w, figsize=(4 * n_w, 4 * n_h), dpi=200)
    if n_h * n_w > 1:
        ax_list = ax_list.reshape(-1)  # make axis list 1d
        for i in range(len(column_list)):
            plot_two_columns(ax_list[i], data_frame, column_list[i], metric_name)
    else:
        plot_two_columns(ax_list, data_frame, column_list[0], metric_name)

    # hide the unused plots
    for i in range(len(column_list), n_w * n_h):
        ax_list[i].axis('off')
    return fig


def main(global_results_csv, metric, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_df = pd.read_csv(global_results_csv)
    # treat two boolean columns categorically
    results_df['trigger_target_class'] = results_df['trigger_target_class'].astype('category')
    results_df['ground_truth'] = results_df['ground_truth'].astype('category')
    results_df['poisoned'] = results_df['poisoned'].astype('category')

    results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, 0.5)

    # drop columns with only one unique value
    to_drop = list()
    for col in list(results_df.columns):
        if len(results_df[col].unique()) <= 1:
            to_drop.append(col)
    results_df = results_df.drop(columns=to_drop)

    features_list = list(results_df.columns)
    if metric not in features_list:
        raise RuntimeError('Selected metric "{}" is not a valid column in the csv file'.format(metric))
    features_list.remove(metric)

    # plot the primary controlled factors
    primary_factors = ['number_classes', 'trigger_type', 'number_triggered_classes']
    fig = plot_feature_list(results_df, primary_factors, metric)
    plt.savefig(os.path.join(output_dir, 'primary-factors.png'))
    plt.close(fig)

    # plot the robustness factors
    factors_list = list(results_df.columns)
    for name in primary_factors:
        factors_list.remove(name)
    factors_list.remove(metric)
    fig = plot_feature_list(results_df, factors_list, metric)

    plt.savefig(os.path.join(output_dir, 'robustness-factors.png'))
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

