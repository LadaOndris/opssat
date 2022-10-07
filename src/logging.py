import os
from datetime import datetime


def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_timestamped_dir(path: str, suffix: str = '') -> str:
    """
    Creates a new directory with a name of a current timestamp
    in a location defined by 'path'.
    Parameters
    ----------
    path    string : Location of the new directory.
    Returns
    -------
    Returns path to the new directory.
    """
    timestamp = get_current_timestamp()
    subdir = os.path.join(path, timestamp + suffix)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    return subdir


def make_log_dir(path: str, suffix: str = '') -> str:
    """
    Creates a new directory with a timestamp in the logs directory.
    """
    make_dir(path)
    return make_timestamped_dir(path, suffix)


def make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def compose_ckpt_path(log_dir: str):
    ckpt_folder = os.path.join(log_dir, 'train_ckpts')
    if not os.path.isdir(ckpt_folder):
        os.makedirs(ckpt_folder)
    return os.path.join(ckpt_folder, 'weights.{epoch:02d}.h5')
