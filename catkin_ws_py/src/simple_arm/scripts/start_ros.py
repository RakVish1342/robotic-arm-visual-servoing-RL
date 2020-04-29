import os
import signal
from subprocess import Popen
import time


def run(cmd, stdout, stderr):
    """Run a given `cmd` in a subprocess, write logs to stdout / stderr.

    Parameters
    ----------
    cmd : list of str
        Command to run.
    stdout : str or subprocess.PIPE object
        Destination of stdout output.
    stderr : str or subprocess.PIPE object
        Destination of stderr output.

    Returns
    -------
    A subprocess.Popen instance.
    """
    return Popen(cmd, stdout=stdout, stderr=stderr, shell=False,
                 preexec_fn=os.setsid)


def get_stdout_stderr(typ, datetime, dir):
    """Create stdout / stderr file paths."""
    out = '%s_%s_stdout.log' % (datetime, typ)
    err = '%s_%s_stderr.log' % (datetime, typ)
    return os.path.join(dir, out), os.path.join(dir, err)


def check_files_exist(files):
    """Check if given list of files exists.

    Parameters
    ----------
    files : list of str
        Files to check for existence.

    Returns
    -------
    None if all files exist. Else raises a ValueError.
    """
    errors = []
    for f in files:
        if not os.path.exists(f):
            errors.append(f)
    if errors:
        raise ValueError('File does not exist: %s' % errors)


def start_process(cmd, typ, start_time, dpath_logs):
    """Start a subprocess with the given command `cmd`.

    Parameters
    ----------
    cmd : list of str
        Command to run.
    typ : str
        Type of subprocess. This will be included in the logs' file names.
    start_time : str
        Datetime string, will be included in the logs' file names as well as
        the resulting bag's name.
    dpath_logs :
        Path to log direcotry.

    Returns
    -------
    A subprocess.Popen instance.
    """
    print('Starting', typ.upper())
    stdout, stderr = get_stdout_stderr(typ, start_time, dpath_logs)
    with open(stdout, 'wb') as out, open(stderr, 'wb') as err:
        return run(cmd, stdout=out, stderr=err)


def main(args):
    dpath_logs = args.dpath_logs
    script_node = args.script_node
    check_files_exist([script_node, dpath_logs])

    start_time = time.strftime('%Y%m%d_%H%M%S')

    p_ros_core = start_process(['/opt/ros/kinetic/bin/roscore'],
                                'ros', start_time, dpath_logs)

    session_talker_node = start_process(['/bin/bash', script_node],
                               'talker_node', start_time,
                               dpath_logs)

    # print pids in case something goes wrong
    print('PGID ROS: ', os.getpgid(p_ros_core.pid))
    print('PGID TALKER NODE: ', os.getpgid(session_talker_node.pid))


    time.sleep(5)
    print('Killing ROS and talker node.')
    os.killpg(os.getpgid(p_ros_core.pid), signal.SIGTERM)
    os.killpg(os.getpgid(session_talker_node.pid), signal.SIGTERM)


if __name__ == '__main__':
    """Start ROS and the talker node each as a subprocess.

    Examples
    --------

    python  start_ros.py --script_node /notebooks/workspace/talker.sh \
    -l /notebooks/workspace/src/scripts
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--script_node', type=str,
                        default='/notebooks/workspace/talker.sh')
    parser.add_argument('--dpath_logs', '-l', type=str,
                        default='/notebooks/workspace/src/scripts')
    args = parser.parse_args()
    main(args)