import configargparse
import json
from collections import OrderedDict
import os

from leaderboards.trojai_config import TrojaiConfig
from leaderboards.submission import SubmissionManager
from leaderboards.leaderboard import Leaderboard
from leaderboards.actor import ActorManager

class JSONConfigFileParser(configargparse.ConfigFileParser):
    def get_syntax_description(self):
        return ["Config file syntax alled based on JSON format"]

    def parse(self, stream):
        try:
            parsed_obj = json.load(stream)
        except Exception as e:
            raise(configargparse.ConfigFileParserException("Couldn't parse config file: %s" % e))

        result = OrderedDict()
        for key, value in parsed_obj.items():
            if isinstance(value, list):
                result[key] = value
            elif value is None:
                pass
            else:
                result[key] = str(value)

        return result

    def serialize(self, items):
        items = dict(items)
        return json.dumps(items, indent=2, sort_keys=True)


def build_round_csvs(trojai_config: TrojaiConfig, leaderboard_names: list, output_dirpath: str, overwrite_csv=True):
    actor_manager = ActorManager.load_json(trojai_config)

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    if len(leaderboard_names) == 0:
        # Process both archived and active rounds
        active_rounds = trojai_config.active_leaderboard_names
        archive_rounds = trojai_config.archive_leaderboard_names

        leaderboard_names.extend(active_rounds)
        leaderboard_names.extend(archive_rounds)


    for leaderboard_name in leaderboard_names:
        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        leaderboard.generate_metadata_csv(overwrite_csv)
        submission_manager = SubmissionManager.load_json(leaderboard)
        submission_manager.generate_round_results_csv(leaderboard, actor_manager, overwrite_csv)


if __name__ == '__main__':

    parser = configargparse.ArgumentParser(
        config_file_parser_class=JSONConfigFileParser,
        description='Builds the CSV files related to a round. Two CSVs are output, METADATA CSV describing the training and a RESULTS CSV describing the results of the round'
    )
    parser.add_argument('--config-filepath', is_config_file=True, help='The filepath to the config file.')
    parser.add_argument('--trojai-config-filepath', type=str, help='The file path to the trojai config', required=True)
    parser.add_argument('--leaderboard-names', nargs='*', help='The names of leaderboards to use, by default will use those specified in trojai config', default=[])
    parser.add_argument('--save-config-filepath', type=str, help='The path to save the config file.')
    parser.add_argument('--output-dirpath', type=str, help='Output dirpath for CSVs.', required=True)

    args = parser.parse_args()

    if args.save_config_filepath is not None:
        parser.write_config_file(args, [args.save_config_filepath])

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    build_round_csvs(trojai_config, args.leaderboard_names, args.output_dirpath)