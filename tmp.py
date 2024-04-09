# import os
import numpy as np
from leaderboards import metrics

# # Grouped_ROC_AUC
# roc_auc = metrics.ROC_AUC()
#
# ifp = '/home/mmajurski/Downloads/tmp/TrinitySRITrojAI/20231206T212009-submission/20231206T212009-execute'
# # ifp = '/home/mmajurski/Downloads/tmp/Perspecta-PurdueRutgers/20231130T041017-submission/20231130T041017-execute'
# tgt_ifp = '/home/mmajurski/Downloads/tmp/test-dataset/models'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-') and fn.endswith('.txt')]
# fns.sort()
#
# pred = list()
# tgt = list()
# for fn in fns:
#     with open(os.path.join(ifp, fn), 'r') as fh:
#         lines = fh.readlines()
#     pred.append(float(lines[0].strip()))
#
#     tgt_fp = os.path.join(tgt_ifp, fn.replace('.txt', ''), 'ground_truth.csv')
#     with open(tgt_fp, 'r') as fh:
#         lines = fh.readlines()
#     tgt.append(float(lines[0].strip()))
#
# pred = np.asarray(pred)
# tgt = np.asarray(tgt)
#
# roc_auc.compute(pred, tgt, model_names=None, metadata_df=None, actor_name="Perspecta", leaderboard_name="APK", data_split_name='Test', submission_epoch_str='20231130T053128', output_dirpath='/home/mmajurski/Downloads/tmp')


# import os
# from leaderboards.actor import Actor, ActorManager
# from leaderboards.submission import Submission, SubmissionManager
# from leaderboards.leaderboard import Leaderboard, TrojaiConfig
# import shutil
#
# # trojai_config_filepath = '/home/mmajurski/Downloads/trojai-leaderboard/trojai_config.json'
# trojai_config_filepath = '/mnt/isgnas/deploy/trojai/multi-round-leaderboard/trojai_config.json'
# leaderboard_name = 'cyber-network-c2-mar2024'
#
# trojai_config = TrojaiConfig.load_json(trojai_config_filepath)
# actor_manager = ActorManager.load_json(trojai_config)
#
# leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
#
# submission_manager = SubmissionManager.load_json(leaderboard)
# new_sub_manager = SubmissionManager(leaderboard_name)

# for actor in actor_manager.get_actors():
#     print(actor.name)
#     submissions = submission_manager.get_submissions_by_actor(actor)
#     for sub in submissions:
#         print(sub.submission_epoch)
#         new_sub = None
#         if sub.data_split_name == 'sts':
#             # remove the submission folder, as we are not preserving it
#             if os.path.exists(sub.actor_submission_dirpath):
#                 shutil.rmtree(sub.actor_submission_dirpath)
#             else:
#                 print("missing sts submission: {}".format(sub.actor_submission_dirpath))
#         else:
#             new_sub = Submission(g_file=sub.g_file, actor=actor, leaderboard=leaderboard, data_split_name=sub.data_split_name, submission_epoch=sub.submission_epoch, provenance=sub.provenance)
#             new_sub_manager.add_submission(actor, new_sub)
#
# new_sub_manager.save_json(leaderboard)
# print('done')


import os
from leaderboards.actor import Actor, ActorManager
from leaderboards.submission import Submission, SubmissionManager
from leaderboards.leaderboard import Leaderboard, TrojaiConfig
import shutil

trojai_config_filepath = '/home/mmajurski/Downloads/trojai-leaderboard/trojai_config.json'
# trojai_config_filepath = '/mnt/isgnas/deploy/trojai/multi-round-leaderboard/trojai_config.json'
leaderboard_name = 'cyber-network-c2-mar2024'

trojai_config = TrojaiConfig.load_json(trojai_config_filepath)
actor_manager = ActorManager.load_json(trojai_config)

leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)

submission_manager = SubmissionManager.load_json(leaderboard)

for actor in actor_manager.get_actors():
    submissions = submission_manager.get_submissions_by_actor(actor)
    for sub in submissions:

        new_sub = None
        if sub.data_split_name == 'sts':
            # remove the submission folder, as we are not preserving it
            if os.path.exists(sub.actor_submission_dirpath):
                shutil.rmtree(sub.actor_submission_dirpath)
            else:
                print("missing sts submission: {}".format(sub.actor_submission_dirpath))
        else:
            if leaderboard_name in sub.g_file.name:
                print(actor.name)
                print(sub.submission_epoch)
                sub.g_file.name = sub.g_file.name.replace('cyber-network-c2-mar2024', 'cyber-network-c2-feb2024')

submission_manager.save_json(leaderboard)
print('done')

