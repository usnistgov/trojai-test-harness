import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_science import utils


def plot_two_columns(ax, results_df, x_column_name, y_column_name, y_axis_logscale=True):
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
        if len(categories) > 4:
            plt.xticks(rotation=45)
        # ax.set_title(x_column_name)


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
    fig = plt.figure(figsize=(16, 9), dpi=100)
    for name in features_list:
        plt.clf()
        ax = plt.gca()
        plot_two_columns(ax, results_df, name, metric)
        plt.savefig(os.path.join(output_dir, '{}.png'.format(name)))
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot each column of the global results csv file.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--metric', type=str, default='cross_entropy', help='Which column to use as the y-axis')
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    main(args.global_results_csv_filepath, args.metric, args.output_dirpath)

