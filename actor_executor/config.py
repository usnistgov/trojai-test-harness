import json_io


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
        self.slurm_queue = slurm_queue
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
        msg += 'log_file_byte_limit = "{}")'.format(self.log_file_byte_limit)
        return msg

    def save_json(self, filepath: str):
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath: str):
        return json_io.read(filepath)

