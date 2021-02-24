# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

from actor_executor import json_io


class Config(object):
    def __init__(self, actor_json_file: str,
                 submissions_json_file: str,
                 log_file: str,
                 submission_dir: str,
                 execute_window: int,
                 ground_truth_dir: str,
                 html_repo_dir: str,
                 models_dir: str,
                 embedding_dir: str,
                 tokenizer_dir: str,
                 sentiment_classification_dir: str,
                 results_dir: str,
                 token_pickle_file: str,
                 slurm_script_file: str,
                 job_table_name: str,
                 result_table_name: str,
                 vms: dict,
                 slurm_queue: str,
                 evaluate_script: str,
                 log_file_byte_limit: int = None,
                 loss_criteria: float = None):
        self.actor_json_file = actor_json_file
        self.submissions_json_file = submissions_json_file
        self.log_file = log_file
        self.submission_dir = submission_dir
        self.execute_window = execute_window
        self.ground_truth_dir = ground_truth_dir
        self.html_repo_dir = html_repo_dir
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.embedding_dir = embedding_dir
        self.tokenizer_dir = tokenizer_dir
        self.sentiment_classification_dir = sentiment_classification_dir
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
        self.loss_criteria = loss_criteria

    def __str__(self):
        msg = 'Config: (actor_json_file = "{}"\n'.format(self.actor_json_file)
        if hasattr(self, 'submissions_json_file'):
            msg += '\tsubmissions_json_file = "{}"\n'.format(self.submissions_json_file)
        if hasattr(self, 'submission_dir'):
            msg += '\tsubmission_dir = "{}"\n'.format(self.submission_dir)
        if hasattr(self, 'log_file'):
            msg += '\tlog_file = "{}"\n'.format(self.log_file)
        if hasattr(self, 'execute_window'):
            msg += '\texecute_window = "{}"\n'.format(self.execute_window)
        if hasattr(self, 'ground_truth_dir'):
            msg += '\tground_truth_dir = "{}"\n'.format(self.ground_truth_dir)
        if hasattr(self, 'html_repo_dir'):
            msg += '\thtml_repo_dir = "{}"\n'.format(self.html_repo_dir)
        if hasattr(self, 'results_dir'):
            msg += '\tresults_dir = "{}"\n'.format(self.results_dir)
        if hasattr(self, 'tokenizer_dir'):
            msg += '\ttokenizer_dir = "{}"\n'.format(self.tokenizer_dir)
        if hasattr(self, 'embedding_dir'):
            msg += '\tembedding_dir = "{}"\n'.format(self.embedding_dir)
        if hasattr(self, 'sentiment_classification_dir'):
            msg += '\tsentiment_classification_dir = "{}"\n'.format(self.sentiment_classification_dir)
        if hasattr(self, 'models_dir'):
            msg += '\tmodels_dir = "{}"\n'.format(self.models_dir)
        if hasattr(self, 'token_pickle_file'):
            msg += '\ttoken_pickle_file = "{}"\n'.format(self.token_pickle_file)
        if hasattr(self, 'slurm_script_file'):
            msg += '\tslurm_script_file = "{}"\n'.format(self.slurm_script_file)
        if hasattr(self, 'accepting_submissions'):
            msg += '\taccepting_submissions = "{}"\n'.format(self.accepting_submissions)
        if hasattr(self, 'job_table_name'):
            msg += '\tjob_table_name = "{}"\n'.format(self.job_table_name)
        if hasattr(self, 'result_table_name'):
            msg += '\tresult_table_name = "{}"\n'.format(self.result_table_name)
        if hasattr(self, 'vms'):
            msg += '\tvms = "{}"\n'.format(self.vms)
        if hasattr(self, 'slurm_queue'):
            msg += '\tslurm_queue = "{}"\n'.format(self.slurm_queue)
        if hasattr(self, 'evaluate_script'):
            msg += '\tevaluate_script: = "{}"\n'.format(self.evaluate_script)
        if hasattr(self, 'log_file_byte_limit'):
            msg += '\tlog_file_byte_limit = "{}"\n'.format(self.log_file_byte_limit)
        if hasattr(self, 'loss_criteria'):
            msg += '\tloss_criteria = "{}")'.format(self.loss_criteria)
        return msg

    def save_json(self, filepath: str):
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath: str):
        return json_io.read(filepath)
