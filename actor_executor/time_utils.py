import datetime
import re


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


def convert_to_epoch(time_str: str) -> int:
    """
    Convert an ISO time-date string into an epoch number.
    :param time_str: An iso time-date string
    :return int: unix epoch (seconds since Jan 1st 1970 (UTC))
    """
    time = datetime.datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
    time = time.replace(tzinfo=datetime.timezone.utc)
    return int(time.timestamp())


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