import json_io

class HoldoutConfig(object):
    def __init__(self,
                 log_file: str,
                 round_config_filepath: str,
                 holdout_model_dir: str,
                 slurm_queue: str,
                 min_loss_criteria: float,
                 holdout_result_dir: str,
                 evaluate_script: str,
                 slurm_script:str,
                 python_executor_script: str
                 ):
        self.log_file = log_file
        self.round_config_filepath = round_config_filepath
        self.holdout_model_dir = holdout_model_dir
        self.slurm_queue = slurm_queue
        self.min_loss_criteria = min_loss_criteria
        self.holdout_result_dir = holdout_result_dir
        self.evaluate_script = evaluate_script
        self.slurm_script = slurm_script
        self.python_executor_script = python_executor_script


    def __str__(self):
        msg = 'HoldoutConfig: (log_file = "{}"\n'.format(self.log_file)
        msg += 'round_config_filepath: = "{}"\n'.format(self.round_config_filepath)
        msg += 'holdout_model_dir: = "{}"\n'.format(self.holdout_model_dir)
        msg += 'slurm_queue: = "{}"\n'.format(self.slurm_queue)
        msg += 'min_loss_criteria: = "{}"\n'.format(self.min_loss_criteria)
        msg += 'holdout_result_dir: = "{}"\n'.format(self.holdout_result_dir)
        return msg

    def save_json(self, filepath: str):
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath:str):
        return json_io.read(filepath)
