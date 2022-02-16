import os
import pandas as pd


def main(test_global_resutls_csv_filepath, holdout_global_resutls_csv_filepath, output_global_resutls_csv_filepath):
    test_df = pd.read_csv(test_global_resutls_csv_filepath)
    holdout_df = pd.read_csv(holdout_global_resutls_csv_filepath)

    output_df = pd.concat([test_df, holdout_df])

    parent, fn = os.path.split(output_global_resutls_csv_filepath)
    if not os.path.exists(parent):
        os.makedirs(parent)

    output_df.to_csv(output_global_resutls_csv_filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to merge the test and holdout csv files into one.')
    parser.add_argument('--test-global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the test global results data.')
    parser.add_argument('--holdout-global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the holdout global results data.')
    parser.add_argument('--output-global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath to the output global results data.')

    args = parser.parse_args()
    main(args.test_global_results_csv_filepath, args.holdout_global_results_csv_filepath, args.output_global_results_csv_filepath)

