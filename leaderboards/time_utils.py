# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import datetime
import re
import matplotlib.dates as mdate


def convert_epoch_to_iso(epoch: float) -> str:
    """
    Convert a Unix Epoch number into a ISO time-date string.
    :param epoch: A float
    :return str: ISO time-date string
    """
    return datetime.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%dT%H:%M:%S")


def convert_epoch_to_psudo_iso(epoch: float) -> str:
    """
    Convert a Unix Epoch number into a pseudo ISO time-date string.
    :param epoch: A float
    :return str: pseudo ISO time-date string with slightly modified formatting ('-' and ':' have been removed)
    """
    # not actual iso format, '-' and ':' have been removed
    # datetime.datetime.utcfromtimestamp(epoch).isoformat()
    return datetime.datetime.utcfromtimestamp(epoch).strftime("%Y%m%dT%H%M%S")


def get_current_epoch() -> int:
    """
    Get the current Unix Epoch
    :return int: unix epoch (seconds since Jan 1st 1970 (UTC))
    """
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp())


def convert_to_epoch_from_psudo(time_str: str) -> int:
    """
    Converts Psudo ISO time-date string into an epoch number.
    :param time_str: The ISO time-date string from convert_epoch_to_psudo_iso
    :return int: unix epoch (seconds since Jan 1st 1970 (UTC))
    """
    time = datetime.datetime.strptime(time_str, "%Y%m%dT%H%M%S")
    time = time.replace(tzinfo=datetime.timezone.utc)
    return int(time.timestamp())

def convert_to_epoch(time_str: str) -> int:
    """
    Convert an ISO time-date string into an epoch number.
    :param time_str: An iso time-date string
    :return int: unix epoch (seconds since Jan 1st 1970 (UTC))
    """
    time = datetime.datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
    time = time.replace(tzinfo=datetime.timezone.utc)
    return int(time.timestamp())


def convert_to_datetime(time_str: str) -> datetime.datetime:
    """
    Convert an ISO time-date string into a datetime object.
    :param time_str: An iso time-date string
    :return int: unix epoch (seconds since Jan 1st 1970 (UTC))
    """
    time = datetime.datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
    time = time.replace(tzinfo=datetime.timezone.utc)
    return time

def convert_seconds_to_dhms(number_of_seconds: float):
    """
       Convert a number of seconds into days, hours, minutes, and seconds.
       :param number_of_seconds: the number of seconds
       :return int: days
       :return int: hours
       :return int: minutes
       :return int: seconds
       """
    number_of_seconds = int(number_of_seconds)
    days = number_of_seconds // 86400
    number_of_seconds = number_of_seconds % 86400
    hours = number_of_seconds // 3600
    minutes = (number_of_seconds % 3600) // 60
    seconds = (number_of_seconds % 60)
    days = int(days)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return days, hours, minutes, seconds


timeRegex = re.compile(r'^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)?((?P<seconds>[\.\d]+?)s)?$')


def parse_time(time_str: str) -> int:
    """
    Parse a time string e.g. (2h13m) into a timedelta object.

    Modified from virhilo's answer at https://stackoverflow.com/a/4628148/851699

    :param time_str: A string identifying a duration.  (eg. 2h13m)
    :return int: number of seconds
    """
    parts = timeRegex.match(time_str)
    assert parts is not None, "Could not parse any time information from '{}'.  Examples of valid strings: '8h', '2d8h5m20s', '2m4s'".format(time_str)
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return int(datetime.timedelta(**time_params).total_seconds())