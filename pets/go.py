'''
Utils for interfacing with Go
'''
import sys
from subprocess import run

GO_MODULE='.'

def go_run(*args: str):
    '''
    Build (if necessary) and
    run the Go module with provided command-line args
    '''
    cmd = ['go', 'run', GO_MODULE, '--', *args]
    print('Running', *cmd)

    run(cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


if  __name__ == "__main__":
    go_run(*sys.argv[1:])
