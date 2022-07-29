# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
from typing import KeysView

from trojai_leaderboard import json_io
from trojai_leaderboard import slurm
from trojai_leaderboard import time_utils
from trojai_leaderboard.leaderboard import Leaderboard
from trojai_leaderboard.trojai_config import TrojaiConfig


class Actor(object):
    def __init__(self, trojai_config: TrojaiConfig, email: str, name: str, poc_email: str):
        self.email = email
        self.name = name
        self.poc_email = poc_email

        self.last_execution_epochs = {}
        self.last_file_epochs = {}

        self.general_file_status = 'None'

        self.job_statuses = {}
        self.file_statuses = {}

        for leaderboard_name in trojai_config.active_leaderboard_names:
            for dataset_split_name in Leaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES:
                self.reset_leaderboard_submission(leaderboard_name, dataset_split_name)

        self.disabled = False

    def update_all_job_status(self, value, check_value=None):
        for key, value in self.job_statuses.items():
            if check_value is None:
                self.job_statuses[key] = value
            elif check_value == value:
                self.job_statuses[key] = value

    def has_job_status(self, check_value):
        for value in self.job_statuses.values():
            if check_value == value:
                return True
        return False

    def get_last_file_epoch(self, leaderboard_name, data_split_name):
        return self.last_file_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)]

    def get_last_execution_epoch(self, leaderboard_name, data_split_name):
        return self.last_execution_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)]

    def update_job_status(self, leaderboard_name, data_split_name, value):
        self.job_statuses[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def update_file_status(self, leaderboard_name, data_split_name, value):
        self.file_statuses[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def update_last_execution_epoch(self, leaderboard_name, data_split_name, value):
        self.last_execution_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def update_last_file_epoch(self, leaderboard_name, data_split_name, value):
        self.last_file_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def get_leaderboard_key(self, leaderboard_name, data_split_name):
        return '{}_{}'.format(leaderboard_name, data_split_name)

    def _has_leaderboard_metadata(self, leaderboard_name, data_split_name):
        leaderboard_key = self.get_leaderboard_key(leaderboard_name, data_split_name)
        return leaderboard_key in self.last_file_epochs and leaderboard_key in self.last_execution_epochs \
               and leaderboard_key in self.job_statuses and leaderboard_key in self.file_statuses

    def reset_leaderboard_submission(self, leaderboard_name, data_split_name):
        print('Resetting {} for leaderboard: {} and data split {}'.format(self.email, leaderboard_name, data_split_name))
        leaderboard_key = self.get_leaderboard_key(leaderboard_name, data_split_name)
        self.last_execution_epochs[leaderboard_key] = 0
        self.last_file_epochs[leaderboard_key] = 0
        self.job_statuses[leaderboard_key] = 'None'
        self.file_statuses[leaderboard_key] = 'None'

    def __str__(self):
        msg = 'Actor: (\n'
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def save_json(self, filepath: str) -> None:
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath: str) -> 'Actor':
        return json_io.read(filepath)

    def in_queue(self, queue_name) -> bool:
        stdout, stderr = slurm.squeue(self.name, queue_name)
        stdout = str(stdout).split("\\n")
        return len(stdout) > 2

    def is_disabled(self):
        return self.disabled

    def can_submit_time_window(self, leaderboard_name, dataset_split_name, execute_window_seconds, cur_epoch) -> bool:
        # Check if the actor is allowed to execute again or not
        last_execution_epoch = self.last_execution_epochs[self.get_leaderboard_key(leaderboard_name, dataset_split_name)]
        if last_execution_epoch + execute_window_seconds <= cur_epoch:
            return True
        return False

    def get_jobs_table_entry(self, leaderboard_name, dataset_split_name, execute_window, current_epoch: int):

        leaderboard_key = self.get_leaderboard_key(leaderboard_name, dataset_split_name)

        # Check if this is the first time we've encountered this leaderboard
        if not self._has_leaderboard_metadata(leaderboard_name, dataset_split_name):
            self.reset_leaderboard_submission(leaderboard_name, dataset_split_name)

        remaining_time = 0
        if self.last_execution_epochs[leaderboard_key] + execute_window > current_epoch:
            remaining_time = (self.last_execution_epochs[leaderboard_key] + execute_window) - current_epoch

        days, hours, minutes, seconds = time_utils.convert_seconds_to_dhms(remaining_time)
        time_str = "{} d, {} h, {} m, {} s".format(days, hours, minutes, seconds)

        if self.last_file_epochs[leaderboard_key] == 0:
            last_file_timestamp = "None"
        else:
            last_file_timestamp = time_utils.convert_epoch_to_iso(self.last_file_epochs[leaderboard_key])
        if self.last_execution_epochs[leaderboard_key] == 0:
            last_execution_timestamp = "None"
        else:
            last_execution_timestamp = time_utils.convert_epoch_to_iso(self.last_execution_epochs[leaderboard_key])

        return [self.name, last_execution_timestamp, self.job_statuses[leaderboard_key], self.file_statuses[leaderboard_key], last_file_timestamp, time_str]


class ActorManager(object):
    def __init__(self):
        self.actors = dict()

    def __str__(self):
        msg = "Actors: \n"
        for actor in self.actors:
            msg = msg + "  " + actor.__str__() + "\n"
        return msg

    def get_keys(self) -> KeysView:
        return self.actors.keys()

    def add_actor(self, trojai_config: TrojaiConfig, email: str, name: str, poc_email: str) -> None:
        if email in self.actors.keys():
            raise RuntimeError("Actor already exists in ActorManager: {}".format(email))
        for key in self.actors.keys():
            if name == self.actors[key].name:
                raise RuntimeError("Actor Name already exists in ActorManager: {}".format(name))
        created_actor = Actor(trojai_config, email, name, poc_email)
        self.actors[email] = created_actor
        print('Created: {}'.format(created_actor))

    def remove_actor(self, email) -> None:
        if email in self.actors.keys():
            del self.actors[email]
            print('Removed {} from actor manager'.format(email))
        else:
            raise RuntimeError('Invalid key in ActorManager: {}'.format(email))

    def get(self, email) -> Actor:
        if email in self.actors:
            return self.actors[email]
        else:
            raise RuntimeError('Invalid key in ActorManager: {}'.format(email))

    def save_json(self, trojai_config: TrojaiConfig) -> None:
        json_io.write(trojai_config.actors_filepath, self)

    @staticmethod
    def init_file(trojai_config: TrojaiConfig) -> None:
        # Create the json file if it does not exist already
        if not os.path.exists(trojai_config.actors_filepath):
            actor_dict = ActorManager()
            actor_dict.save_json(trojai_config)

    @staticmethod
    def load_json(trojai_config: TrojaiConfig) -> 'ActorManager':
        # make sure the file exists
        ActorManager.init_file(trojai_config)
        return json_io.read(trojai_config.actors_filepath)

    def get_jobs_table(self, leaderboard_name, dataset_split_name, execute_window, cur_epoch: int):
        jobs_table = list()
        for actor in self.actors.values():
            jobs_table.append(actor.get_jobs_table_entry(leaderboard_name, dataset_split_name, execute_window, cur_epoch))
        return jobs_table

    def convert_to_csv(self, output_file):
        write_header = True

        with open(output_file, 'w') as file:
            for actorKey in self.get_keys():
                actor = self.get(actorKey)
                if write_header:
                    for key in actor.__dict__.keys():
                        file.write(key + ',')
                    file.write('\n')
                    write_header = False

                for key in actor.__dict__.keys():
                    file.write(str(actor.__dict__[key]) + ',')
                file.write('\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Creates an actor, and adds it to the actors.json')
    parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)

    parser.add_argument('--add-actor', type=str,
                        help='Adds an actor in CSV "ActorTeamName,ActorTeamEmail,PocEmail", this will exit after adding the actor',
                        default=None)
    parser.add_argument("--remove-actor", type=str,
                        help='Removes an actor (based on actor team name), this will exit after removing the actor (the repo that is owned by the actor will not be deleted)',
                        default=None)
    parser.add_argument("--reset-actor", type=str,
                        help='Resets a team to allow them to resubmit in CSV '
                             '"ActorTeamEmail,leaderboard_name,data_split_name", '
                             'or "ActorTeamEmail,leaderboard_name" for all data split in leaderboard, '
                             'or "ActorTeamEmail for all leaderboards and data splits',
                        default=None)
    parser.add_argument('--convert-to-csv', type=str,
                        help='The file to save actors to CSV (converts json to CSV)',
                        default=None)

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    actor_manager = ActorManager.load_json(trojai_config)

    if args.add_actor is not None:
        items = args.add_actor.split(',')
        if len(items) != 3:
            raise RuntimeError('Invalid number of CSV arguments for add-actor')

        team_name = items[0]
        email = items[1]
        poc_email = items[2]

        try:
            team_name = team_name.encode('ascii')
        except:
            raise RuntimeError('Team name must be ASCII only')

        team_name = team_name.decode()
        team_name = str(team_name)
        invalid_chars = [" ", "/", ">", "<", "|", ":", "&", ",", ";", "?", "\\", "*"]
        for char in invalid_chars:
            if char in team_name:
                raise RuntimeError('team_name cannot have invalid characters: {}'.format(invalid_chars))

        actor_manager.add_actor(trojai_config, email, team_name, poc_email)


    elif args.reset_actor is not None:
        items = args.reset_actor.split(',')

        data_splits = []
        leaderboards = []

        if len(items) == 3:
            email = items[0]
            leaderboards.append(items[1])
            data_splits.append(items[2])
        elif len(items) == 2:
            email = items[0]
            leaderboards.append(items[1])
            data_splits.extend(Leaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES)
        elif len(items) == 1:
            email = items[0]
            leaderboards.extend(trojai_config.active_leaderboard_names)
            data_splits.extend(Leaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES)
        else:
            raise RuntimeError('Invalid number of CSV arguments for reset-actor')

        actor = actor_manager.get(email)

        for l_name in leaderboards:
            for data_split_name in data_splits:
                actor.reset_leaderboard_submission(l_name, data_split_name)

    elif args.remove_actor is not None:
        actor_manager.remove_actor(args.remove_actor)
    elif args.convert_to_csv is not None:
        actor_manager.convert_to_csv(args.convert_to_csv)
        exit(0)

    actor_manager.save_json(trojai_config)
