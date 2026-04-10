#!/usr/bin/env -S python -u 

# import sys
# sys.path.append('../../SRC')

import os
import shutil

cases = [ ]
Re = 0.5 * 2.54 / 100 # 0.5 in => meters

import numpy as np
# AR = [ 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2 ]
AR = np.geomspace(0.01, 2, 50)
for aspect_ratio in AR:
    cases.append( { 'Re':Re , 'H':Re*aspect_ratio})   

template_file = 'template.yaml'
runs_dir = 'runs'

with open(template_file, 'r') as f:
    template = f.read()

if os.path.exists(runs_dir):
    shutil.rmtree(runs_dir)
os.makedirs(runs_dir)

for i, case in enumerate(cases):
    run_dir = f'{runs_dir}/run.{i}'
    os.makedirs(run_dir)

    tags_str = ','.join([f'{k}:{v}' for k, v in list(case.items())[:3]])
    desc = f'run{i}_{tags_str}'

    case["run_dir"] = run_dir
    case["run_id"] = i
    case["tags"] = tags_str
    case["desc"] = desc

    print(case)
    config = template
    for tag, value in case.items():
        config = config.replace(f'%{tag}%', str(value))

    with open(f'{run_dir}/config.yaml', 'w') as f:
        f.write(config)

print(f'Generated {len(cases)} cases in {runs_dir}/')
