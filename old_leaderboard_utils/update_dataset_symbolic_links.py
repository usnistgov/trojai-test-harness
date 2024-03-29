import os
from leaderboards.trojai_config import TrojaiConfig

def main(args):
    dataset_dirpath = args.dataset_dirpath
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    old_round_names = args.old_round_names
    new_round_names = args.new_round_names
    split_names = args.split_names
    trojai_datasets_dirpath = trojai_config.datasets_dirpath
    use_round_name_prefix = args.use_round_name_prefix

    for i, round_name in enumerate(old_round_names):
        trojai_round_dataset_dirpath = os.path.join(trojai_datasets_dirpath, new_round_names[i])

        round_dataset_dirpath = os.path.join(dataset_dirpath, round_name)

        if not os.path.exists(round_dataset_dirpath):
            print('Unable to find path: {}'.format(round_dataset_dirpath))
            continue
        print('Creating links for {}'.format(round_name))
        os.makedirs(trojai_round_dataset_dirpath, exist_ok=True)

        for split_name in split_names:
            link_name = split_name
            if '_' in link_name:
                link_name = link_name.replace('_', '-')

            if 'tokenizers' in split_name or 'source_data' in split_name:
                dataset_name = split_name
            else:
                if use_round_name_prefix:
                    dataset_name = '{}-{}'.format(round_name, split_name)
                else:
                    dataset_name = split_name

            source = os.path.join(round_dataset_dirpath, dataset_name)
            if not os.path.exists(source):
                print('Unable to find source: {}'.format(source))
                continue

            link_filepath = os.path.join(trojai_round_dataset_dirpath, link_name)

            if os.path.exists(link_filepath):
                print('Link path already exists: {}'.format(link_filepath))
                continue

            os.symlink(source, link_filepath)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Utility to creates symbolic links to the datasets')
    parser.add_argument('--dataset-dirpath', type=str, help='The main dataset directory path')
    parser.add_argument('--trojai-config-filepath', type=str, help='The filepath the main trojai config')
    parser.add_argument('--old-round-names', nargs='+', help='The names of the rounds')
    parser.add_argument('--new-round-names', nargs='+', help='The names to map the old round names too')
    parser.add_argument('--split-names', nargs='+', help='The dataset split names')
    parser.add_argument('--use-round-name-prefix', action='store_true', help='The prefix to be added to the split name')

    args = parser.parse_args()

    main(args)