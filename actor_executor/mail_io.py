import os
import smtplib

import json_io
import time_utils


class TrojaiMail(object):
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

