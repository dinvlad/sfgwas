'''
Utils for interfacing with Go
'''
import os
import sys
from subprocess import run

GO_EXEC='./go-pets'

def _run(*cmd: str):
    '''
    Execute a command with arguments
    '''
    print('Running', *cmd)
    run(cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )

def go_run(*args: str):
    '''
    Build (if necessary) and
    run the Go executable with provided command-line args
    '''
    if not os.path.exists(GO_EXEC):
        _run('go', 'build', '-o', GO_EXEC)

    _run(GO_EXEC, *args)


if  __name__ == "__main__":
    go_run(*sys.argv[1:])
