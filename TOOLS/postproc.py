#!/usr/bin/env python3
"""Generic post-processing dispatcher.

Calls py/postproc.py in the current demo directory, forwarding all arguments.
Run from any demo directory that contains a py/postproc.py script.

USAGE (from a demo directory):
    ./postproc.py                   # run py/postproc.py
    ./postproc.py --max-time 500    # forward args to the demo script
    ./postproc.py --script py/plot_pq.py  # run a different script in py/
"""

import argparse
import sys
import subprocess
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _TOOLS_DIR.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False)
    parser.add_argument("--script", default="py/postproc.py",
                        help="Script to run relative to cwd (default: py/postproc.py)")
    args, remaining = parser.parse_known_args()

    script = Path.cwd() / args.script
    if not script.exists():
        print(f"Error: script not found: {script}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(script)] + remaining
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
