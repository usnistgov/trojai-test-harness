import logging
import pytablewriter
import fcntl
import traceback
from pytablewriter import HtmlTableWriter
from git import Repo
from git.exc import GitCommandError

from actor_executor.actor import Actor, ActorManager
from actor_executor.submission import Submission, SubmissionManager
import actor_executor.time_utils
from actor_executor import slurm
from actor_executor.mail_io import TrojaiMail



def update_html(html_dir: str, actor_manager: ActorManager, submission_manager: SubmissionManager, execute_window: int,
                job_table_name: str, result_table_name: str, push_html: bool, cur_epoch: int, accepting_submissions: bool, slurm_queue: str):
    lock_filepath = "/var/lock/htmlpush-lockfile"
    with open(lock_filepath, mode='w') as fh_lock:
        try:
            fcntl.lockf(fh_lock, fcntl.LOCK_EX)

            # Populate results table
            if submission_manager is not None:
                scores = submission_manager.get_score_table()

                scoreWriter = HtmlTableWriter()
                scoreWriter.headers = ["Team", "Cross Entropy", "CE 95% CI", "Brier Score", "ROC-AUC", "Runtime (s)", "Execution Timestamp", "File Timestamp", "Parsing Errors", "Launch Errors"]
                scoreWriter.value_matrix = scores
                scoreWriter.type_hints = [pytablewriter.String, # Team
                                          pytablewriter.RealNumber, # Cross Entropy
                                          pytablewriter.RealNumber, # CE 95% CI
                                          pytablewriter.RealNumber, # Brier Score
                                          pytablewriter.RealNumber, # ROC-AUC
                                          pytablewriter.Integer,  # Runtime
                                          pytablewriter.String, # Execution Timestamp
                                          pytablewriter.String, # File Timestamp
                                          pytablewriter.String, # arsing Errors
                                          pytablewriter.String]  # Launch Errors
                scoreTable = scoreWriter.dumps()


                for line in scoreTable.splitlines():
                    if "<th>" in line:
                        newLine = line.replace("<th>", "<th class=\"th-sm\">")
                        scoreTable = scoreTable.replace(line, newLine)
                    if "<table" in line:
                        newLine = line.replace("<table", "<table id=\"" + result_table_name + "\" class=\"table table-striped table-bordered table-sm\" cellspacing=\"0\" width=\"100%\"")
                        scoreTable = scoreTable.replace(line, newLine)


                scoreTableHtml = """
                <!-- ******RESULTS****** -->    
                <div class="table-responsive">
                """ + scoreTable + """
                </div>   
                """

                with open(html_dir + "/_includes/" + result_table_name + ".html", mode='w', encoding='utf-8') as f:
                    f.write(scoreTableHtml)

            if actor_manager is not None:
                # Populate jobs table
                activeJobs = actor_manager.get_jobs_table(execute_window, cur_epoch)

                activeJobWriter = HtmlTableWriter()
                activeJobWriter.headers = ["Team", "Execution Timestamp", "Job Status", "File Status", "File Timestamp",
                                           "Time until next execution"]
                activeJobWriter.type_hints = [pytablewriter.String, pytablewriter.String, pytablewriter.String,
                                              pytablewriter.String, pytablewriter.String, pytablewriter.String]

                activeJobWriter.value_matrix = activeJobs
                jobTable = activeJobWriter.dumps()

                for line in jobTable.splitlines():
                    if "<th>" in line:
                        newLine = line.replace("<th>", "<th class=\"th-sm\">")
                        jobTable = jobTable.replace(line, newLine)
                    if "<table" in line:
                        newLine = line.replace("<table", "<table id=\"" + job_table_name + "\" class=\"table table-striped table-bordered table-sm\" cellspacing=\"0\" width=\"100%\"")
                        jobTable = jobTable.replace(line, newLine)

                jobTableHtml = """
                <!-- ******JOBS****** -->    
                <div class="table-responsive">
                """ + jobTable + """
                </div>   
                """

                with open(html_dir + "/_includes/" + job_table_name + ".html", mode='w', encoding='utf-8') as f:
                    f.write(jobTableHtml)

            # Push the HTML to the web
            if push_html:
                repo = Repo(html_dir)
                if repo.is_dirty() or not accepting_submissions:

                    timestampUpdate = """
                    var uploadTimestamp = """ + str(cur_epoch) + """;
                    var d = new Date(0);
                    d.setUTCSeconds(uploadTimestamp);
                    var acceptingSubmissions = """ + str(accepting_submissions).lower() + """; 
                    
                    $(document).ready(function () {
                        $('#timestamp').text(d.toISOString().split('.')[0] );
                    });
                    """

                    with open(html_dir + "/js/time-updater.js", mode='w', encoding='utf-8') as f:
                        f.write(timestampUpdate)

                idleNodes = int(slurm.sinfo_node_query(slurm_queue, "idle"))
                mixNodes = int(slurm.sinfo_node_query(slurm_queue, "mix"))
                drainingNodes = int(slurm.sinfo_node_query(slurm_queue, "draining"))
                runningNodes = mixNodes + drainingNodes
                downNodes = int(slurm.sinfo_node_query(slurm_queue, "down"))
                drainedNodes = int(slurm.sinfo_node_query(slurm_queue, "drained"))
                if downNodes > 0:
                    msg = '{} SLURM Node(s) Down in queue {}'.format(str(downNodes), slurm_queue)
                    TrojaiMail().send('trojai@nist.gov', msg, msg)

                webDownNodes = downNodes + drainedNodes

                acceptingSubmissionsUpdate = """
                var """ + slurm_queue + """AcceptingSubmission = """ + str(accepting_submissions).lower() + """;
                var """ + slurm_queue + """IdleNodes = """ + str(idleNodes) + """;
                var """ + slurm_queue + """RunningNodes = """ + str(runningNodes) + """;
                var """ + slurm_queue + """DownNodes = """ + str(webDownNodes) + """;
                
                 $(document).ready(function () {
                        $('#""" + slurm_queue + """IdleNodes').text(""" + slurm_queue + """IdleNodes);
                        $('#""" + slurm_queue + """RunningNodes').text(""" + slurm_queue + """RunningNodes);
                        $('#""" + slurm_queue + """DownNodes').text(""" + slurm_queue + """DownNodes);
                        $('#""" + slurm_queue + """AcceptingSubmission').text(""" + slurm_queue + """AcceptingSubmission);
                    });
                """

                with open(html_dir + "/js/" + slurm_queue + "-submission.js", mode='w', encoding='utf-8') as f:
                    f.write(acceptingSubmissionsUpdate)

                git = repo.git()
                try:
                    git.pull()

                    gitAddList = list()
                    if actor_manager is not None:
                        gitAddList.append(html_dir + "/_includes/" + job_table_name + ".html")
                    if submission_manager is not None:
                        gitAddList.append(html_dir + "/_includes/" + result_table_name + ".html")

                    gitAddList.append(html_dir + "/js/time-updater.js")
                    gitAddList.append(html_dir + "/js/" + slurm_queue + "-submission.js")

                    git.add(gitAddList)
                    git.commit("-m", "Actor update {}".format(
                        time_utils.convert_epoch_to_psudo_iso(cur_epoch)))

                    git.push()
                    logging.info("Web-site content has been pushed.")
                except GitCommandError as ex:
                    if "nothing to commit" not in str(ex):
                        logging.error("Git had errors when trying to commit and push: " + str(ex))
                    else:
                        logging.info("Commit to repo not pushed, no changes found")
            else:
                logging.info("Web push disabled")

        except:
            msg = 'html_output threw an exception, releasing file lock "{}" regardless.{}'.format(lock_filepath, traceback.format_exc())
            TrojaiMail().send('trojai@nist.gov', 'html_output fallback lockfile release', msg)
            raise
        finally:
            fcntl.lockf(fh_lock, fcntl.LOCK_UN)

