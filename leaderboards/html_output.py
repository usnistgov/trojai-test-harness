# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging
import fcntl
import traceback
import os
from git import Repo
from git.exc import GitCommandError
from airium import Airium

from leaderboards.actor import  ActorManager
from leaderboards.submission import SubmissionManager
from leaderboards import time_utils
from leaderboards import slurm
from leaderboards.mail_io import TrojaiMail
from leaderboards.trojai_config import TrojaiConfig
from leaderboards.leaderboard import Leaderboard

def update_html_pages(trojai_config: TrojaiConfig, active_leaderboards_dict: dict, active_submission_managers_dict: dict, commit_and_push: bool):
    cur_epoch = time_utils.get_current_epoch()

    lock_filepath = "/var/lock/htmlpush-lockfile"
    with open(lock_filepath, mode='w') as fh_lock:
        try:
            fcntl.lockf(fh_lock, fcntl.LOCK_EX)

            active_leaderboards = []
            for leaderboard_name, leaderboard in active_leaderboards_dict.items():
                active_leaderboards.append(leaderboard)

            active_leaderboards.sort(key=lambda x: x.html_leaderboard_priority, reverse=True)

            html_dirpath = trojai_config.html_repo_dirpath

            html_output_dirpath = os.path.join(html_dirpath, '_includes')

            written_files = []

            leaderboards_filepath = os.path.join(html_output_dirpath, 'leaderboards.html')

            a = Airium()
            html_default_leaderboard = trojai_config.html_default_leaderboard_name
            if html_default_leaderboard == '':
                if len(active_leaderboards) > 0:
                    html_default_leaderboard = active_leaderboards[0].name
            with a.ul(klass='nav nav-pills', id='leaderboardTabs', role='tablist'):
                with a.li(klass='nav-item'):
                    for leaderboard in active_leaderboards:
                        if html_default_leaderboard == leaderboard.name:
                            a.a(klass='nav-link waves-light active show', id='tab-{}'.format(leaderboard.name), href='#{}'.format(leaderboard.name), **{'data-toggle': 'tab', 'aria-controls': '{}'.format(leaderboard.name), 'aria-selected': 'true'}, _t=leaderboard.name)
                        else:
                            a.a(klass='nav-link waves-light', id='tab-{}'.format(leaderboard.name), href='#{}'.format(leaderboard.name), **{'data-toggle': 'tab', 'aria-controls': '{}'.format(leaderboard.name), 'aria-selected': 'false'}, _t=leaderboard.name)

            with a.div(klass='tab-content card'):
                for leaderboard in active_leaderboards:
                    a('{{% include {}-leaderboard.html %}}'.format(leaderboard.name))

            with open(leaderboards_filepath, 'w') as f:
                f.write(str(a))

            written_files.append(leaderboards_filepath)

            # Check for existance of about files
            for leaderboard in active_leaderboards:
                filepath = os.path.join(html_output_dirpath, 'about-{}.html'.format(leaderboard.name))
                if not os.path.exists(filepath):
                    a = Airium()
                    with a.div(klass='card-body card-body-cascade text-center pb-0'):
                        with a.p(klass='card-text text-left'):
                            a('Placeholder text for {}'.format(leaderboard.name))
                        with a.div(klass='container'):
                            a.img(src='public/images/trojaiLogo.png', klass='img-fluid', alt='Placeholder image')
                        with a.p():
                            a('Placeholder image description')

                    with open(filepath, 'w') as f:
                        f.write(str(a))
                    written_files.append(filepath)

            actor_manager = ActorManager.load_json(trojai_config)

            for leaderboard in active_leaderboards:
                submission_manager = active_submission_managers_dict[leaderboard.name]

                is_first = False
                if html_default_leaderboard == leaderboard.name:
                    is_first = True

                filepath = leaderboard.write_html_leaderboard(html_output_dirpath, is_first)
                written_files.append(filepath)

                for data_split_name in leaderboard.get_all_data_split_names():
                    execute_window = leaderboard.get_submission_window_time(data_split_name)
                    filepath = actor_manager.write_jobs_table(html_output_dirpath, leaderboard.name, data_split_name, execute_window, cur_epoch)
                    written_files.append(filepath)
                    filepath = submission_manager.write_score_table_unique(html_output_dirpath, leaderboard, data_split_name)
                    written_files.append(filepath)
                    filepath = submission_manager.write_score_table(html_output_dirpath, leaderboard, data_split_name)
                    written_files.append(filepath)


            table_javascript_filepath = os.path.join(trojai_config.html_repo_dirpath, 'js', 'trojai-table-init.js')

            content = """
$(document).ready(function () {

var sort_col;\n
"""

            # configure javascript for table controls
            for leaderboard in active_leaderboards:
                html_sort_options = leaderboard.html_table_sort_options
                for key, info_dict in html_sort_options.items():
                    column_name = info_dict['column']
                    order = info_dict['order']
                    content += """sort_col = $('#{}').find("th:contains('{}')")[0].cellIndex;\n""".format(key, column_name)
                    content += "$('#{}').dataTable({{ order: [[ sort_col, '{}' ]] }});\n\n".format(key, order)

            content += "$('.dataTables_length').addClass('bs-select');\n});"
            with open(table_javascript_filepath, 'w') as f:
                f.write(content)

            written_files.append(table_javascript_filepath)
            # Push the HTML to the web
            repo = Repo(html_dirpath)
            if repo.is_dirty() or not trojai_config.accepting_submissions:
                timestampUpdate = """
var uploadTimestamp = """ + str(cur_epoch) + """;
var d = new Date(0);
d.setUTCSeconds(uploadTimestamp);
var acceptingSubmissions = """ + str(trojai_config.accepting_submissions).lower() + """; 

$(document).ready(function () {
   $('#timestamp').text(d.toISOString().split('.')[0] );
});
                   """

                time_update_filepath = os.path.join(html_dirpath, 'js', 'time-updater.js')
                with open(time_update_filepath, mode='w', encoding='utf-8') as f:
                    f.write(timestampUpdate)

                written_files.append(time_update_filepath)

            # TODO: Update
            allocatedNodes = 1 # int(slurm.sinfo_node_query(slurm_queue, "alloc"))
            idleNodes = 1 # int(slurm.sinfo_node_query(slurm_queue, "idle"))
            mixNodes = 1 # int(slurm.sinfo_node_query(slurm_queue, "mix"))
            drainingNodes = 1 #int(slurm.sinfo_node_query(slurm_queue, "draining"))
            runningNodes = allocatedNodes  # "alloc" should include mix and draining
            downNodes = 1 # int(slurm.sinfo_node_query(slurm_queue, "down"))
            drainedNodes = 1 # int(slurm.sinfo_node_query(slurm_queue, "drained"))

            # TODO: Update
            # if downNodes > 0:
            #     msg = '{} SLURM Node(s) Down in queue {}'.format(str(downNodes), slurm_queue)
            #     TrojaiMail().send('trojai@nist.gov', msg, msg)

            webDownNodes = downNodes + drainedNodes

            slurm_queue = 'sts'
            acceptingSubmissionsUpdate = """
var """ + slurm_queue + """AcceptingSubmission = """ + str(trojai_config.accepting_submissions).lower() + """;
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

            slurm_submission_filepath = os.path.join(html_dirpath, 'js', '{}-submissions.js'.format(slurm_queue))

            with open(slurm_submission_filepath, mode='w', encoding='utf-8') as f:
                f.write(acceptingSubmissionsUpdate)
            written_files.append(slurm_submission_filepath)

            git = repo.git()
            try:
                git.pull()
                git.add(written_files)

                if commit_and_push:
                    git.commit("-m", "Actor update {}".format(time_utils.convert_epoch_to_psudo_iso(cur_epoch)))
                    git.push()
                    logging.info("Web-site content has been pushed.")
            except GitCommandError as ex:
                if "nothing to commit" not in str(ex):
                    logging.error("Git had errors when trying to commit and push: " + str(ex))
                else:
                    logging.info("Commit to repo not pushed, no changes found")


        except:
            msg = 'html_output threw an exception, releasing file lock "{}" regardless.{}'.format(lock_filepath,
                                                                                                  traceback.format_exc())
            TrojaiMail().send('trojai@nist.gov', 'html_output fallback lockfile release', msg)
            raise
        finally:
            fcntl.lockf(fh_lock, fcntl.LOCK_UN)