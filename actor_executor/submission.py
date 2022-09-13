# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import logging

from actor_executor import json_io



class Submission(object):
    def __init__(self,  submission_dirpath: str, results_dirpath: str, ground_truth_dirpath: str, slurm_queue: str):
        self.file = None
        self.actor = None

        self.slurm_queue = slurm_queue
        self.cross_entropy = None
        self.cross_entropy_95_confidence_interval = None
        self.roc_auc = None
        self.brier_score = None
        self.execution_runtime = None
        self.model_execution_runtimes = None
        self.execution_epoch = None
        self.slurm_job_name = None
        self.slurm_output_filename = None
        self.confusion_output_filename = None
        self.web_display_parse_errors = "None"
        self.web_display_execution_errors = "None"

        self.ground_truth_dirpath = ground_truth_dirpath

        # create the directory where submissions are stored
        self.global_submission_dirpath = submission_dirpath
        if not os.path.isdir(os.path.join(self.global_submission_dirpath, self.actor.name)):
            logging.info("Submission directory for " + self.actor.name + " does not exist, creating ...")
            os.makedirs(os.path.join(self.global_submission_dirpath, self.actor.name))

        # create the directory where results are stored
        self.global_results_dirpath = results_dirpath
        if not os.path.isdir(os.path.join(self.global_results_dirpath, self.actor.name)):
            logging.info("Results directory for " + self.actor.name + " does not exist, creating ...")
            os.makedirs(os.path.join(self.global_results_dirpath, self.actor.name))

    def __str__(self) -> str:
        msg = 'file name: "{}", from eamil: "{}"'.format(self.file.name, self.actor.email)
        return msg

class SubmissionManager(object):
    def __init__(self):
        # keyed on email
        self.__submissions = dict()

    def submissions(self):
        return self.__submissions

    def __str__(self):
        msg = ""
        for a in self.__submissions.keys():
            msg = msg + "Actor: {}: \n".format(a)
            submissions = self.__submissions[a]
            for s in submissions:
                msg = msg + "  " + s.__str__() + "\n"
        return msg

    @staticmethod
    def load_json(filepath: str):
        return json_io.read(filepath)
