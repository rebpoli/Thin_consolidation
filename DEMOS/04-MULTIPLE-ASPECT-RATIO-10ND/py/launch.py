#!/usr/bin/env -S python -u 

import subprocess
import os
import time
from pathlib import Path

max_parallel = 8
runs_dir = 'runs'
run_script = 'run.py'

run_dirs = sorted([d for d in Path(runs_dir).iterdir() if d.is_dir() and d.name.startswith('run.')])

running = {}
completed = []
pending = list(run_dirs)
start_time = time.time()

while pending or running:
    for pid, info in list(running.items()):
        if info['process'].poll() is not None:
            elapsed = time.time() - info['start_time']
            completed.append({'dir': info['dir'], 'desc': info['desc'], 'time': elapsed})
            print(f"Completed: {info['desc']} ({elapsed:.1f}s)")
            del running[pid]

    while len(running) < max_parallel and pending:
        run_dir = pending.pop(0)
        desc_file = run_dir / 'description.txt'
        desc = desc_file.read_text().strip() if desc_file.exists() else run_dir.name

        process = subprocess.Popen(
            ['python', f'../../{run_script}'],
            cwd=str(run_dir),
            stdout=open(run_dir / 'stdout.log', 'w'),
            stderr=open(run_dir / 'stderr.log', 'w')
        )

        running[process.pid] = {
            'process': process,
            'dir': run_dir,
            'desc': desc,
            'start_time': time.time()
        }
        print(f"Started: {desc} (PID {process.pid})")

    time.sleep(0.5)

total_time = time.time() - start_time
avg_time = sum(c['time'] for c in completed) / len(completed) if completed else 0
min_time = min(c['time'] for c in completed) if completed else 0
max_time = max(c['time'] for c in completed) if completed else 0

print(f"\n{'='*60}")
print(f"Summary: {len(completed)} cases completed")
print(f"Total walltime: {total_time:.1f}s")
print(f"Average runtime: {avg_time:.1f}s (min: {min_time:.1f}s, max: {max_time:.1f}s)")
print(f"{'='*60}")

# import subprocess
# import os
# import time
# from pathlib import Path

# max_parallel = 10
# runs_dir = 'runs'
# run_script = 'run.py'

# run_dirs = sorted([d for d in Path(runs_dir).iterdir() if d.is_dir() and d.name.startswith('run.')])

# running = {}
# completed = []
# pending = list(run_dirs)

# while pending or running:
#     for pid, info in list(running.items()):
#         if info['process'].poll() is not None:
#             completed.append(info['dir'])
#             print(f"Completed: {info['dir'].name}")
#             del running[pid]

#     while len(running) < max_parallel and pending:
#         run_dir = pending.pop(0)
#         desc_file = run_dir / 'description.txt'
#         desc = desc_file.read_text().strip() if desc_file.exists() else run_dir.name

#         process = subprocess.Popen(
#             ['python', f'../../{run_script}'],
#             cwd=str(run_dir),
#             stdout=open(run_dir / 'stdout.log', 'w'),
#             stderr=open(run_dir / 'stderr.log', 'w')
#         )

#         running[process.pid] = {'process': process, 'dir': run_dir, 'desc': desc}
#         print(f"Started: {desc} (PID {process.pid})")

#     time.sleep(0.5)

# print(f"All {len(completed)} cases completed")
