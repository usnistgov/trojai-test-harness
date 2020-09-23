from actor_executor import json_io


class Config(object):
    def __init__(self, actor_json_file: str,
                 submissions_json_file: str,
                 log_file: str,
                 submission_dir: str,
                 execute_window: int,
                 ground_truth_dir: str,
                 html_repo_dir: str,
                 results_dir: str,
                 token_pickle_file: str,
                 slurm_script_file: str,
                 job_table_name: str,
                 result_table_name: str,
                 vms: dict,
                 slurm_queue: str,
                 evaluate_script: str,
                 log_file_byte_limit: int):
        self.actor_json_file = actor_json_file
        self.submissions_json_file = submissions_json_file
        self.log_file = log_file
        self.submission_dir = submission_dir
        self.execute_window = execute_window
        self.ground_truth_dir = ground_truth_dir
        self.html_repo_dir = html_repo_dir
        self.results_dir = results_dir
        self.token_pickle_file = token_pickle_file
        self.slurm_script_file = slurm_script_file
        self.accepting_submissions = True
        self.job_table_name = job_table_name
        self.result_table_name = result_table_name
        self.vms = vms
        self.evaluate_script = evaluate_script
        self.slurm_queue = slurm_queue
        # TODO log file byte limit should be set to None by default, and it should be supported by the codebase (its currently commented out)
        self.log_file_byte_limit = log_file_byte_limit

    def __str__(self):
        msg = 'Config: (actor_json_file = "{}"\n'.format(self.actor_json_file)
        msg += 'submissions_json_file = "{}"\n'.format(self.submissions_json_file)
        msg += 'submission_dir = "{}"\n'.format(self.submission_dir)
        msg += 'log_file = "{}"\n'.format(self.log_file)
        msg += 'execute_window = "{}"\n'.format(self.execute_window)
        msg += 'ground_truth_dir = "{}"\n'.format(self.ground_truth_dir)
        msg += 'html_repo_dir = "{}"\n'.format(self.html_repo_dir)
        msg += 'results_dir = "{}"\n'.format(self.results_dir)
        msg += 'token_pickle_file = "{}"\n'.format(self.token_pickle_file)
        msg += 'slurm_script_file = "{}"\n'.format(self.slurm_script_file)
        msg += 'accepting_submissions = "{}"\n'.format(self.accepting_submissions)
        msg += 'job_table_name = "{}"\n'.format(self.job_table_name)
        msg += 'result_table_name = "{}"\n'.format(self.result_table_name)
        msg += 'vms = "{}"\n'.format(self.vms)
        msg += 'slurm_queue = "{}"\n'.format(self.slurm_queue)
        msg += 'evaluate_script: = "{}"\n'.format(self.evaluate_script)
        msg += 'log_file_byte_limit = "{}")'.format(self.log_file_byte_limit)
        return msg

    def save_json(self, filepath: str):
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath: str):
        return json_io.read(filepath)


class HoldoutConfig(object):
    def __init__(self,
                 log_file: str,
                 round_config_filepath: str,
                 model_dir: str,
                 slurm_queue: str,
                 min_loss_criteria: float,
                 results_dir: str,
                 evaluate_script: str,
                 slurm_script: str,
                 python_executor_script: str,
                 submission_dir: str
                 ):
        self.log_file = log_file
        self.round_config_filepath = round_config_filepath
        self.model_dir = model_dir
        self.slurm_queue = slurm_queue
        self.min_loss_criteria = min_loss_criteria
        self.results_dir = results_dir
        self.evaluate_script = evaluate_script
        self.slurm_script = slurm_script
        # TODO what are the differences between the python_executor script and the evaluate_script
        self.python_executor_script = python_executor_script
        self.submission_dir = submission_dir

    def __str__(self):
        msg = 'HoldoutConfig: (log_file = "{}"\n'.format(self.log_file)
        msg += 'round_config_filepath: = "{}"\n'.format(self.round_config_filepath)
        msg += 'model_dir: = "{}"\n'.format(self.model_dir)
        msg += 'slurm_queue: = "{}"\n'.format(self.slurm_queue)
        msg += 'min_loss_criteria: = "{}"\n'.format(self.min_loss_criteria)
        msg += 'results_dir: = "{}"\n'.format(self.results_dir)
        msg += 'evaluate_script: = "{}"\n'.format(self.evaluate_script)
        msg += 'slurm_script: = "{}"\n'.format(self.slurm_script)
        msg += 'python_executor_script: = "{}"\n'.format(self.python_executor_script)
        msg += 'submission_dir: = "{}"\n'.format(self.submission_dir)
        return msg

    def save_json(self, filepath: str):
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath:str):
        return json_io.read(filepath)