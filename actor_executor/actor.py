import os
from typing import KeysView

import json_io
import slurm
import time_utils


class Actor(object):
    def __init__(self, email: str, name: str, poc_email: str):
        self.email = email
        self.name = name
        self.poc_email = poc_email
        self.last_execution_epoch = 0
        self.last_file_epoch = 0
        self.job_status = "None"
        self.file_status = "None"
        self.disabled = False

    def __str__(self):
        msg = "(name: {} email: {} poc: {} last_execution_epoch: {} last_file_epoch: {} job_status: {} file_status: {} disabled: {})".format(self.name, self.email, self.poc_email, self.last_execution_epoch, self.last_file_epoch, self.job_status, self.file_status, self.disabled)
        return msg

    def save_json(self, filepath: str) -> None:
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath: str):
        return json_io.read(filepath)

    def reset(self) -> None:
        self.last_execution_epoch = 0
        self.last_file_epoch = 0
        self.file_status = "None"
        self.job_status = "Reset"

    def in_queue(self, queue_name) -> bool:
        stdout, stderr = slurm.squeue(self.name, queue_name)
        stdout = str(stdout).split("\\n")
        return len(stdout) > 2

    def is_disabled(self):
        return self.disabled

    def can_submit_timewindow(self, execute_window_seconds, cur_epoch) -> bool:
        # Check if the actor is allowed to execute again or not
        if self.last_execution_epoch + execute_window_seconds <= cur_epoch:
            return True
        return False

    def get_jobs_table_entry(self, execute_window, current_epoch: int):

        remaining_time = 0
        if self.last_execution_epoch + execute_window > current_epoch:
            remaining_time = (self.last_execution_epoch + execute_window) - current_epoch

        days, hours, minutes, seconds = time_utils.convert_seconds_to_dhms(remaining_time)
        time_str = "{} d, {} h, {} m, {} s".format(days, hours, minutes, seconds)

        if self.last_file_epoch == 0:
            last_file_timestamp = "None"
        else:
            last_file_timestamp = time_utils.convert_epoch_to_iso(self.last_file_epoch)
        if self.last_execution_epoch == 0:
            last_execution_timestamp = "None"
        else:
            last_execution_timestamp = time_utils.convert_epoch_to_iso(self.last_execution_epoch)

        return [self.name, last_execution_timestamp, self.job_status, self.file_status, last_file_timestamp, time_str]


class ActorManager(object):
    def __init__(self):
        self.__actors = dict()

    def __str__(self):
        msg = "Actors: \n"
        for actor in self.__actors:
            msg = msg + "  " + actor.__str__() + "\n"
        return msg

    def get_keys(self) -> KeysView:
        return self.__actors.keys()

    def add_actor(self, email: str, name: str, poc_email: str) -> None:
        if email in self.__actors.keys():
            raise RuntimeError("Actor already exists in ActorManager")
        for key in self.__actors.keys():
            if name == self.__actors[key].name:
                raise RuntimeError("Actor Name already exists in ActorManager")
        self.__actors[email] = Actor(email, name, poc_email)

    def remove_actor(self, email) -> None:
        if email in self.__actors:
            del self.__actors[email]

    def get(self, email) -> Actor:
        if email in self.__actors:
            return self.__actors[email]
        else:
            raise RuntimeError('Invalid key in ActorManager')

    def save_json(self, filepath: str) -> None:
        json_io.write(filepath, self)

    @staticmethod
    def init_file(filepath: str) -> None:
        # Create the json file if it does not exist already
        if not os.path.exists(filepath):
            actor_dict = ActorManager()
            actor_dict.save_json(filepath)

    @staticmethod
    def load_json(filepath: str):
        # make sure the file exists
        ActorManager.init_file(filepath)
        return json_io.read(filepath)

    def get_jobs_table(self, execute_window, cur_epoch: int):
        jobs_table = list()
        for key in self.__actors.keys():
            jobs_table.append(self.__actors[key].get_jobs_table_entry(execute_window, cur_epoch))
        return jobs_table


