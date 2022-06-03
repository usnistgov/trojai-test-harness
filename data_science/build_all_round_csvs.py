import build_round_csvs
import os
import json

def main(csv_config_dirpath, overwrite_csv=False):
    if not os.path.exists(csv_config_dirpath):
        print('ERROR Unable to find path: {}'.format(csv_config_dirpath))
        return

    for file in os.listdir(csv_config_dirpath):
        if not file.endswith('.json'):
            continue

        with open(os.path.join(csv_config_dirpath, file), 'r') as f:
            config_contents = json.load(f)
            round_dataset_dirpath = config_contents["round-dataset-dirpath"]
            round_results_dirpath = config_contents["round-results-dirpath"]
            output_dirpath = config_contents["output-dirpath"]
            result_dataset_mapping = build_round_csvs.parse_dataset_mapping(config_contents["result-dataset-mapping"])

            build_round_csvs.build_round_csvs(round_dataset_dirpath, round_results_dirpath, output_dirpath,
                                              result_dataset_mapping, overwrite_csv=overwrite_csv)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Builds all round CSVs')
    parser.add_argument('--csv-config-dirpath', type=str, required=True, help='The path to all csv config files for building round CSVS.')
    parser.add_argument('--overwrite-csv', action="store_true")

    args = parser.parse_args()

    main(args.csv_config_dirpath, args.overwrite_csv)



