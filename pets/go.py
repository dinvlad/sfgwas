'''
Utils for interfacing with Go
'''
import sys
from pathlib import Path
from subprocess import run


def _run(cmd: str, *args: str):
    '''
    Execute a command with arguments
    '''
    print('Running:', cmd, *args)
    run([cmd, *args],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def go_run(cmd_path: Path, *args: str):
    '''
    Build (if necessary) and
    run the Go executable with provided command-line args

    Note: the challenge format allows us to pass data
    between stages only over the model and client/server dirs,
    hence we should pass the built executable through it as well.
    '''
    cmd = cmd_path.absolute().as_posix()
    if not cmd_path.exists():
        _run('go', 'build', '-o', cmd)

    _run(cmd, *args)


def path_str(path: Path):
    '''
    Convert Path into an absolute POSIX path string
    '''
    return path.absolute().as_posix()


# example invocation
if  __name__ == "__main__":
    go_run(Path('go-pets'), *sys.argv[1:])
