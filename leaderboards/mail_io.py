# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import smtplib

from leaderboards import json_io
from leaderboards import time_utils


class TrojaiMail(object):

    DEUGGING = False

    SERVER = "smtp.nist.gov"
    FROM = 'trojai@nist.gov'
    CACHE_FILEPATH = '/tmp/trojai-mail-cache.json'

    def __init__(self):
        self.sent_cache = list()

        if not os.path.exists(self.CACHE_FILEPATH):
            self.save_cache()

    def load_cache(self):
        self.sent_cache = json_io.read(self.CACHE_FILEPATH)
        # filter out old cache elements
        stale_timeout = 3600  # 1 hour in seconds
        current_epoch = time_utils.get_current_epoch()
        self.sent_cache = [x for x in self.sent_cache if abs(current_epoch - x[1]) < stale_timeout]

    def save_cache(self):
        json_io.write(self.CACHE_FILEPATH, self.sent_cache)

    def reset_cache(self):
        self.sent_cache = list()
        self.save_cache()

    def send(self, to: str, subject: str, message: str):
        if TrojaiMail.DEUGGING:
            return

        try:
            self.load_cache()

            in_cache = any([x for x in self.sent_cache if subject.lower().strip() == x[0]])
            if not in_cache:
                mail_message = "From: {}\r\nTo: {}\r\nSubject: {}\r\n\r\n\r\n{}".format(self.FROM, to, subject, message)

                # Send the mail
                server = smtplib.SMTP(self.SERVER)
                server.sendmail(self.FROM, to, mail_message)
                server.quit()

                current_epoch = time_utils.get_current_epoch()
                self.sent_cache.append((subject.lower().strip(), current_epoch))
                self.save_cache()
        except:
            pass

