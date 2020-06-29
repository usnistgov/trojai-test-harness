import json_io
import time_utils


class GoogleDriveFile(object):
    def __init__(self, email: str, file_name: str, file_id: str, modified_timestamp: str):
        self.email = email
        self.name = file_name
        self.id = file_id
        self.modified_epoch = time_utils.convert_to_epoch(modified_timestamp)

    def __str__(self):
        msg = 'file id: "{}", name: "{}", modified_epoch: "{}", email: "{}" '.format(self.id, self.name, self.modified_epoch, self.email)
        return msg

    def save_json(self, file_path: str):
        json_io.write(file_path, self)

    @staticmethod
    def load_json(file_path: str):
        return json_io.read(file_path)
