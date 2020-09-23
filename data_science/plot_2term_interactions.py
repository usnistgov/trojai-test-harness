import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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


def compute_mean_effects_matrix(data_frame, factor1_name, factor2_name, metric_name, number_levels, summary_statistic='mean'):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    f1_is_numeric = str(data_frame[factor1_name].dtype) in numerics

    f2_is_numeric = str(data_frame[factor2_name].dtype) in numerics
    f2_vals = data_frame[factor2_name].copy()
    f2_vals = utils.replace_invalid(f2_vals)

    if f2_is_numeric:
        f2_vals = f2_vals.astype(float).to_numpy()
        f2_vals = f2_vals[np.isfinite(f2_vals)]

        f2_kmeans = KMeans(number_levels).fit(f2_vals.reshape(-1, 1))
        f2_vals_unique = np.asarray(f2_kmeans.cluster_centers_).round(decimals=2).reshape(-1)
    else:
        f2_vals_unique = utils.unique_non_null(f2_vals)

    f1_keys = list()
    df_vals_per_f1 = dict()
    if f1_is_numeric:
        tmp_df = data_frame.copy()
        tmp_df = utils.replace_invalid(tmp_df)
        tmp_df = tmp_df[tmp_df[factor1_name].notna()]

        f1_vals = tmp_df[factor1_name].to_numpy().astype(np.float32)
        if np.sum(np.isfinite(f1_vals)) != np.size(f1_vals):
            raise RuntimeError('Missed Nan')

        f1_kmeans = KMeans(number_levels).fit(f1_vals.reshape(-1, 1))
        f1_clusters = f1_kmeans.predict(f1_vals.reshape(-1, 1))
        f1_vals_unique = np.asarray(f1_kmeans.cluster_centers_).round(decimals=2).reshape(-1)

        # create subset of data_frame y_vals that has only the current feature level
        for k1 in range(number_levels):
            tmp_df = tmp_df[f1_clusters == k1]
            f1_keys.append(f1_vals_unique[k1])
            df_vals_per_f1[f1_vals_unique[k1]] = tmp_df
    else:
        tmp_df = data_frame.copy()
        tmp_df = utils.replace_invalid(tmp_df)
        tmp_df = tmp_df[tmp_df[factor1_name].notna()]

        f1_vals = tmp_df[factor1_name]

        f1_vals_unique = utils.unique_non_null(f1_vals)
        for c in f1_vals_unique:
            tmp_df = tmp_df[tmp_df[factor1_name] == c]
            f1_keys.append(c)
            df_vals_per_f1[c] = tmp_df

    means_matrix = None
    support_matrix = None
    for f1_key in f1_keys:
        tmp_df = df_vals_per_f1[f1_key]
        y = np.zeros(0, dtype=np.float64)
        y_counts = np.zeros(0, dtype=np.float64)

        if f2_is_numeric:
            f2_df = tmp_df.copy()
            f2_df = utils.replace_invalid(f2_df)
            f2_df = f2_df[f2_df[factor2_name].notna()]

            vals = f2_df[factor2_name].to_numpy().astype(np.float32)
            f2_clusters = f2_kmeans.predict(vals.reshape(-1, 1))

            for k in range(number_levels):
                tmp_y = f2_df[f2_clusters == k][metric_name].copy().to_numpy()
                if np.size(tmp_y) == 0:
                    # no support for this factor configuration
                    y = np.append(y, np.nan)
                    y_counts = np.append(y_counts, 0)
                else:
                    if summary_statistic is 'mean':
                        y = np.append(y, np.mean(tmp_y))
                        y_counts = np.append(y_counts, tmp_y.size)
                    elif summary_statistic is 'median':
                        y = np.append(y, np.median(tmp_y))
                        y_counts = np.append(y_counts, tmp_y.size)
                    else:
                        raise RuntimeError('Invalid summary statistic: {}'.format(summary_statistic))
            if means_matrix is None:
                means_matrix = y
                support_matrix = y_counts
            else:
                means_matrix = np.vstack((means_matrix, y))
                support_matrix = np.vstack((support_matrix, y_counts))
        else:
            f2_df = tmp_df.copy()
            f2_df = utils.replace_invalid(f2_df)
            f2_df = f2_df[f2_df[factor2_name].notna()]

            for c in f2_vals_unique:
                tmp_y = f2_df[f2_df[factor2_name] == c][metric_name].copy().to_numpy().astype(np.float32)
                if np.size(tmp_y) == 0:
                    # no support for this factor configuration
                    y = np.append(y, np.nan)
                    y_counts = np.append(y_counts, 0)
                else:
                    if summary_statistic is 'mean':
                        y = np.append(y, np.mean(tmp_y))
                        y_counts = np.append(y_counts, tmp_y.size)
                    elif summary_statistic is 'median':
                        y = np.append(y, np.median(tmp_y))
                        y_counts = np.append(y_counts, tmp_y.size)
                    else:
                        raise RuntimeError('Invalid summary statistic: {}'.format(summary_statistic))
            if means_matrix is None:
                means_matrix = y
                support_matrix = y_counts
            else:
                means_matrix = np.vstack((means_matrix, y))
                support_matrix = np.vstack((support_matrix, y_counts))

    return means_matrix, support_matrix, f1_vals_unique, f2_vals_unique


def compute_mean_effects(data_frame, x_column_name, y_column_name, number_levels, summary_statistic='mean'):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    y_vals = data_frame[y_column_name].copy()
    x_vals = data_frame[x_column_name].copy()
    y_vals = utils.replace_invalid(y_vals)
    y_vals = y_vals.astype(float).to_numpy()
    x_vals = utils.replace_invalid(x_vals)

    if str(x_vals.dtype) in numerics:
        x_vals = x_vals.astype(float).to_numpy()

        kmeans = KMeans(number_levels).fit(x_vals.reshape(-1, 1))
        clusters = kmeans.predict(x_vals.reshape(-1, 1))
        x = kmeans.cluster_centers_
        x = np.asarray(x)
        y = np.zeros(0, dtype=np.float64)
        for k in range(number_levels):
            tmp_y = y_vals[clusters == k]
            tmp_y = tmp_y[np.isfinite(tmp_y)]
            if summary_statistic is 'mean':
                y = np.append(y, np.mean(tmp_y))
            elif summary_statistic is 'median':
                y = np.append(y, np.median(tmp_y))
            else:
                raise RuntimeError('Invalid summary statistic: {}'.format(summary_statistic))
    else:
        x = utils.unique_non_null(x_vals)
        x = np.asarray(x)
        y = np.zeros(0, dtype=np.float64)
        for c in x:
            tmp_y = y_vals[x_vals == c]
            tmp_y = tmp_y[np.isfinite(tmp_y)]
            if summary_statistic is 'mean':
                y = np.append(y, np.mean(tmp_y))
            elif summary_statistic is 'median':
                y = np.append(y, np.median(tmp_y))
            else:
                raise RuntimeError('Invalid summary statistic: {}'.format(summary_statistic))

    return x, y


def main(factor_global_results_csv_filepath, output_dirpath):

    results_df = pd.read_csv(factor_global_results_csv_filepath)
    # treat two boolean columns categorically
    results_df['trigger_target_class'] = results_df['trigger_target_class'].astype('category')
    results_df['ground_truth'] = results_df['ground_truth'].astype('category')
    results_df['poisoned'] = results_df['poisoned'].astype('category')

    columns = list(results_df.columns)
    factors_list = list()
    for c in columns:
        if c.endswith('_level'):
            results_df[c] = results_df[c].astype('category')
            factors_list.append(c.replace('_level', ''))

    # results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, 0.5)

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
            plot_titles = f_vals[factor2]
            n_w, n_h = get_subplot_size(mean_effects_matrix.shape[1])

            # fig, ax_list = plt.subplots(1, n_w, figsize=(4 * n_w, 4), dpi=100)
            fig, ax_list = plt.subplots(n_h, n_w, figsize=(4 * n_w, 4 * n_h), dpi=200)
            ax_list = ax_list.reshape(-1)  # make axis list 1d
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
                ax.set_title('{} - {}'.format(factor2, plot_titles[i]))

                if len(x) >= 4:
                    plt.xticks(rotation=45)
                plt.tight_layout()

            # hide the unused plots
            for i in range(mean_effects_matrix.shape[1], n_w * n_h):
                ax_list[i].axis('off')

            plt.savefig(os.path.join(output_dir, '{}-{}.png'.format(factor1, factor2)))
            plt.close(fig)
        break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot DEX analysis.')
    parser.add_argument('--factor-global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    main(args.factor_global_results_csv_filepath, args.output_dirpath)

