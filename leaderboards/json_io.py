# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import fcntl
import json
import os.path

import jsonpickle
import logging
import traceback

from leaderboards.mail_io import TrojaiMail


def write(filepath, obj):
    if not filepath.endswith('.json'):
        raise RuntimeError("Expecting a file ending in '.json'")
    lock_file = '/var/lock/trojai-json_io-lockfile'
    with open(lock_file, 'w') as lfh:
        try:
            fcntl.lockf(lfh, fcntl.LOCK_EX)

            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(obj, warn=True, indent=2))
        except Exception as ex:
            logging.error(ex)
            msg = 'json_io failed writing file "{}" releasing file lock regardless.{}'.format(filepath, traceback.format_exc())
            TrojaiMail().send('trojai@nist.gov','json_io write fallback lockfile release',msg)
            raise
        finally:
            fcntl.lockf(lfh, fcntl.LOCK_UN)


def read(filepath):
    if not filepath.endswith('.json'):
        raise RuntimeError("Expecting a file ending in '.json'")
    lock_file = '/var/lock/trojai-json_io-lockfile'
    with open(lock_file, 'w') as lfh:
        try:
            fcntl.lockf(lfh, fcntl.LOCK_EX)
            with open(filepath, mode='r', encoding='utf-8') as f:
                obj = jsonpickle.decode(f.read())
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'json_io failed reading file "{}" releasing file lock regardless.{}'.format(filepath, traceback.format_exc())
            TrojaiMail().send('trojai@nist.gov','json_io write fallback lockfile release',msg)
            raise
        finally:
            fcntl.lockf(lfh, fcntl.LOCK_UN)
    return obj

