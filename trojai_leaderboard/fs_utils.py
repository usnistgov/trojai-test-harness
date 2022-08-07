# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import collections
import typing
import logging
import shutil


def truncate_log_file(filepath: str, byte_limit: int):
    # truncate log file to N bytes
    if os.path.exists(filepath) and (byte_limit is not None) and (byte_limit > 0):
        if (1.01 * os.path.getsize(filepath)) > byte_limit:  # use 1% buffer
            shutil.copyfile(filepath, filepath.replace('.txt', '.orig.txt'))
            os.truncate(filepath, byte_limit)

            with open(filepath, 'a') as fh:
                fh.write('\n\n**** Log File Truncated ****\n\n')


def write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds, confusion_filepath):
    with open(confusion_filepath, 'w', newline='\n') as fh:
        fh.write('Threshold, TP, FP, FN, TN, TPR, FPR\n')
        for i in range(len(thresholds)):
            fh.write('{}, {:d}, {:d}, {:d}, {:d}, {}, {}\n'.format(float(thresholds[i]), int(TP_counts[i]), int(FP_counts[i]), int(FN_counts[i]), int(TN_counts[i]), float(TPR[i]), float(FPR[i])))



